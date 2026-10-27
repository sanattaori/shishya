"""SystemOneBackend against a mock HTTP transport: request shape, parsing, errors and retries."""
import json
from types import SimpleNamespace

import httpx
import pytest

from shishya.backends import BackendError, SystemOneBackend
from shishya.backends import systemone as systemone_mod
from shishya.schema import compile_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["approve", "deny", "review"],
            "description": "Decide the request.",
            "x-shishya": {"options": {"approve": "Grant it", "deny": "Refuse it"}},
        },
        "urgent": {"type": "boolean", "description": "Is it urgent?"},
    },
}
COMP = compile_schema(SCHEMA)
VERDICT = COMP.field("verdict")
URGENT = COMP.field("urgent")
OPTIONS = {"verdict": ("approve", "review"), "urgent": (True, False)}


def make_backend(handler, **kw):
    b = SystemOneBackend("http://sys.test/", **kw)
    b._client.close()
    b._client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def json_reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(systemone_mod, "time", SimpleNamespace(sleep=sleeps.append))
    return sleeps


# -- request ------------------------------------------------------------------------------------------

def test_request_body_and_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {
            "verdict": {"probabilities": {"approve": 0.5, "review": 0.5}},
            "urgent": {"noul": 0.5},
        }})

    b = make_backend(handler)
    b.score("Need 5 days off", [VERDICT, URGENT], OPTIONS)

    assert seen["method"] == "POST"
    assert seen["url"] == "http://sys.test/v1/systemone"
    body = seen["body"]
    assert body["state"] == "Need 5 days off"
    assert set(body["questions"]) == {"verdict", "urgent"}


def test_request_question_types_and_criteria():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {
            "verdict": {"probabilities": {"approve": 0.5, "review": 0.5}},
            "urgent": {"noul": 0.5},
        }})

    make_backend(handler).score("text", [VERDICT, URGENT], OPTIONS)
    q = seen["body"]["questions"]

    assert q["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}
    # criteria map each allowed option to its text, or null when the schema gives no text
    assert q["verdict"] == {
        "type": "choice",
        "instructions": "Decide the request.",
        "criteria": {"approve": "Grant it", "review": None},
    }
    assert "deny" not in q["verdict"]["criteria"]    # not an allowed option, so not asked


def test_model_is_sent_only_when_set():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"verdict": {"probabilities": {"approve": 1, "review": 0}}}})

    make_backend(handler).score("t", [VERDICT], OPTIONS)
    make_backend(handler, model="qwen-small").score("t", [VERDICT], OPTIONS)
    assert "model" not in seen[0]
    assert seen[1]["model"] == "qwen-small"


def test_api_key_is_sent_as_bearer_token():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"answers": {"verdict": {"probabilities": {"approve": 1, "review": 0}}}})

    make_backend(handler, api_key="k-123").score("t", [VERDICT], OPTIONS)
    assert seen["auth"] == "Bearer k-123"


def test_no_api_key_means_no_auth_header(monkeypatch):
    monkeypatch.delenv("SHISHYA_BACKEND_KEY", raising=False)
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"answers": {"verdict": {"probabilities": {"approve": 1, "review": 0}}}})

    make_backend(handler).score("t", [VERDICT], OPTIONS)
    assert seen["auth"] is None


# -- parsing ------------------------------------------------------------------------------------------

def test_parses_choice_probabilities():
    b = make_backend(json_reply({"answers": {
        "verdict": {"probabilities": {"approve": 0.7, "review": 0.3}},
    }}))
    got = b.score("t", [VERDICT], OPTIONS)
    assert got == {"verdict": {"approve": 0.7, "review": 0.3}}


def test_parses_noul_with_noul_key():
    b = make_backend(json_reply({"answers": {"urgent": {"noul": 0.8}}}))
    got = b.score("t", [URGENT], OPTIONS)
    assert got["urgent"] == pytest.approx({True: 0.8, False: 0.2})


def test_parses_noul_with_probabilities_true_key():
    b = make_backend(json_reply({"answers": {"urgent": {"probabilities": {"true": 0.25, "false": 0.75}}}}))
    got = b.score("t", [URGENT], OPTIONS)
    assert got["urgent"] == pytest.approx({True: 0.25, False: 0.75})


def test_parses_all_fields_together():
    b = make_backend(json_reply({"answers": {
        "verdict": {"probabilities": {"approve": 0.4, "review": 0.6}},
        "urgent": {"noul": 0.9},
    }}))
    got = b.score("t", [VERDICT, URGENT], OPTIONS)
    assert set(got) == {"verdict", "urgent"}
    assert got["verdict"] == {"approve": 0.4, "review": 0.6}


def test_model_name_from_response_is_recorded_when_not_set():
    b = make_backend(json_reply({"model": "m-7", "answers": {"urgent": {"noul": 0.5}}}))
    b.score("t", [URGENT], OPTIONS)
    assert b.model == "m-7"


# -- errors -------------------------------------------------------------------------------------------

def test_http_422_is_backend_error_without_retry(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(422, text='{"detail":"bad question"}')

    b = make_backend(handler)
    with pytest.raises(BackendError, match="422"):
        b.score("t", [VERDICT], OPTIONS)
    assert len(calls) == 1
    assert no_sleep == []


def test_http_400_is_backend_error_without_retry(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="nope")

    with pytest.raises(BackendError, match="400"):
        make_backend(handler).score("t", [VERDICT], OPTIONS)
    assert len(calls) == 1


def test_valid_json_with_wrong_shape_is_backend_error():
    b = make_backend(json_reply({"result": "oops"}))
    with pytest.raises(BackendError, match="unexpected answer"):
        b.score("t", [VERDICT], OPTIONS)


def test_missing_field_in_answers_is_backend_error():
    b = make_backend(json_reply({"answers": {"verdict": {"probabilities": {"approve": 1, "review": 0}}}}))
    with pytest.raises(BackendError, match="unexpected answer"):
        b.score("t", [VERDICT, URGENT], OPTIONS)


def test_non_numeric_probability_is_backend_error():
    b = make_backend(json_reply({"answers": {"urgent": {"noul": "high"}}}))
    with pytest.raises(BackendError, match="unexpected answer"):
        b.score("t", [URGENT], OPTIONS)


def test_non_json_body_is_backend_error():
    b = make_backend(lambda request: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(BackendError):
        b.score("t", [VERDICT], OPTIONS)


# -- retries ------------------------------------------------------------------------------------------

def test_retry_on_503_then_success(no_sleep):
    replies = [httpx.Response(503, text="busy"),
               httpx.Response(200, json={"answers": {"verdict": {"probabilities": {"approve": 0.6, "review": 0.4}}}})]
    calls = []

    def handler(request):
        calls.append(1)
        return replies.pop(0)

    b = make_backend(handler)
    got = b.score("t", [VERDICT], OPTIONS)
    assert got == {"verdict": {"approve": 0.6, "review": 0.4}}
    assert len(calls) == 2
    assert no_sleep == [0.5]


def test_retry_on_429_then_success(no_sleep):
    replies = [httpx.Response(429, text="slow down"), httpx.Response(200, json={"answers": {"urgent": {"noul": 0.1}}})]
    b = make_backend(lambda request: replies.pop(0))
    assert b.score("t", [URGENT], OPTIONS)["urgent"][True] == pytest.approx(0.1)


def test_retry_on_connection_error_then_success(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"answers": {"urgent": {"noul": 0.3}}})

    b = make_backend(handler)
    assert b.score("t", [URGENT], OPTIONS)["urgent"][True] == pytest.approx(0.3)
    assert len(calls) == 2


def test_gives_up_after_retries(no_sleep):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="down")

    b = make_backend(handler, retries=2)
    with pytest.raises(BackendError, match="failed"):
        b.score("t", [VERDICT], OPTIONS)
    assert len(calls) == 3
    assert no_sleep[:2] == [0.5, 1.0]


def test_no_sleep_after_last_failed_attempt(no_sleep):
    b = make_backend(lambda request: httpx.Response(503, text="down"), retries=2)
    with pytest.raises(BackendError):
        b.score("t", [VERDICT], OPTIONS)
    assert no_sleep == [0.5, 1.0]
