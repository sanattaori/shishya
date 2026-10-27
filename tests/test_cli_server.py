"""CLI, HTTP server, ChatLogprobsBackend and OpenRouterGuru, all without a network."""
from __future__ import annotations

import json
import math

import httpx
import pytest

from shishya.backends import BackendError, ChatLogprobsBackend, FakeBackend
from shishya.cli import main
from shishya.guru import OpenRouterGuru
from shishya.schema import compile_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "deny"], "description": "What to do?",
                    "x-shishya": {"options": {"approve": "Grant it", "deny": "Refuse it"}}},
        "urgent": {"type": "boolean", "description": "Is it urgent?"},
    },
}
RULES = '''
from shishya import rule

@rule(uses=["verdict"])
def never_approve(ans, ctx):
    "Never approve."
    return ans["verdict"] != "approve"
'''


# -- server ---------------------------------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from shishya.server import create_app
    return TestClient(create_app(FakeBackend()))


def test_server_decide(client):
    r = client.post("/v1/decide", json={"input": "hello", "schema": SCHEMA, "min_probability": 0.0})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["answer"]) == {"verdict", "urgent"}


def test_server_bad_schema(client):
    r = client.post("/v1/decide", json={"input": "hello", "schema": {"type": "array"}})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


def test_server_unknown_key(client):
    r = client.post("/v1/decide", json={"input": "hello", "schema": SCHEMA, "rules": "x"})
    assert r.status_code == 422
    assert "rules" in r.json()["error"]["message"]


def test_server_missing_input(client):
    assert client.post("/v1/decide", json={"schema": SCHEMA}).status_code == 422


def test_healthz(client):
    r = client.get("/healthz")
    assert r.json()["ok"] is True and "version" in r.json()


# -- cli ------------------------------------------------------------------------------------------
def test_cli_decide_json(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "rules.py").write_text(RULES)
    code = main(["decide", str(tmp_path / "s.json"), "--input", "please", "--backend", "fake",
                 "--rules", str(tmp_path / "rules.py"), "--facts", '{"a": 1}', "--json"])
    assert code == 0
    d = json.loads(capsys.readouterr().out)
    assert d["status"] == "ok" and d["answer"]["verdict"] == "deny"


def test_cli_decide_summary_and_unsure(tmp_path, capsys):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    code = main(["decide", str(tmp_path / "s.json"), "--input", "please", "--backend", "fake",
                 "--min-probability", "0.999"])
    assert code == 2
    assert "status: unsure" in capsys.readouterr().out


def test_cli_error(tmp_path, capsys):
    code = main(["decide", str(tmp_path / "missing.json"), "--input", "x", "--backend", "fake"])
    assert code == 1
    assert capsys.readouterr().err.startswith("error:")


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip()


# -- ChatLogprobsBackend --------------------------------------------------------------------------
def _logprob_reply(pairs):
    return {"choices": [{"logprobs": {"content": [{"token": pairs[0][0], "logprob": pairs[0][1],
                                                  "top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs]}]}}]}


def test_chat_logprobs_math():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        assert body["max_tokens"] == 1 and body["logprobs"] is True and body["top_logprobs"] == 20
        if "Is it urgent" in body["messages"][0]["content"]:
            return httpx.Response(200, json=_logprob_reply([("B", math.log(0.9)), ("A", math.log(0.1))]))
        return httpx.Response(200, json=_logprob_reply(
            [("A", math.log(0.5)), (" a", math.log(0.1)), ("B", math.log(0.2)), ("the", math.log(0.2))]))

    comp = compile_schema(SCHEMA)
    be = ChatLogprobsBackend(client=httpx.Client(transport=httpx.MockTransport(handler)))
    opts = {f.name: f.options for f in comp.fields}
    out = be.score("text", list(comp.fields), opts)
    assert out["verdict"]["approve"] == pytest.approx(0.6 / 0.8)
    assert out["verdict"]["deny"] == pytest.approx(0.2 / 0.8)
    assert out["urgent"][True] == pytest.approx(0.1) and out["urgent"][False] == pytest.approx(0.9)
    assert len(seen) == 2
    assert "A. approve: Grant it" in seen[0]["messages"][0]["content"]


def test_chat_logprobs_uniform_and_errors():
    comp = compile_schema(SCHEMA)
    f = comp.field("verdict")
    zero = httpx.MockTransport(lambda r: httpx.Response(200, json=_logprob_reply([("x", -1.0)])))
    out = ChatLogprobsBackend(client=httpx.Client(transport=zero)).score("t", [f], {"verdict": f.options})
    assert out["verdict"] == {"approve": 0.5, "deny": 0.5}
    bad = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(BackendError):
        ChatLogprobsBackend(client=httpx.Client(transport=bad)).score("t", [f], {"verdict": f.options})
    with pytest.raises(BackendError):
        ChatLogprobsBackend(client=httpx.Client(transport=zero)).score(
            "t", [f], {"verdict": tuple(f"o{i}" for i in range(27))})


# -- OpenRouterGuru -------------------------------------------------------------------------------
FIELDS = [{"name": "verdict", "kind": "choice", "options": ["approve", "deny"], "instructions": "What to do?",
           "option_text": {"approve": "Grant it"}}]


def _chat(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_guru_parses_fenced_json_and_prompt():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.headers, json.loads(req.content)))
        return _chat('```json\n{"verdict": "deny"}\n```')

    g = OpenRouterGuru(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert g.propose("hi", FIELDS, ["Not allowed: Never approve."]) == {"verdict": "deny"}
    headers, body = seen[0]
    assert headers["x-title"] == "Shishya" and body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    user = body["messages"][1]["content"]
    assert "approve: Grant it" in user and "Not allowed: Never approve." in user


def test_guru_retries_then_fails(monkeypatch):
    monkeypatch.setattr("shishya.guru.time.sleep", lambda s: None)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "1"}, text="slow down") if len(calls) < 3 else _chat("{}")

    g = OpenRouterGuru(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert g.propose("hi", FIELDS, []) == {} and len(calls) == 3
    always = httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
    with pytest.raises(RuntimeError, match="503"):
        OpenRouterGuru(api_key="k", client=httpx.Client(transport=always)).propose("hi", FIELDS, [])


def test_guru_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterGuru().propose("hi", FIELDS, [])
