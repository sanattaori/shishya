"""ClaudeCodeGuru: builds the right command, reads structured output, and its answers are never trainable."""
import json
import subprocess

import pytest

from shishya import Shishya
from shishya.backends import FakeBackend
from shishya.guru import ClaudeCodeGuru

FIELDS = [{"name": "action", "kind": "choice", "options": ["run", "ask", "block"], "instructions": "Run?",
           "option_text": {}},
          {"name": "destructive", "kind": "boolean", "options": [True, False], "instructions": "Destructive?",
           "option_text": {}}]


@pytest.fixture
def fake_claude(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append({"cmd": cmd, **kw})
        out = {"type": "result", "is_error": False, "result": "", "structured_output": {"action": "ask",
                                                                                        "destructive": True}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", run)
    return calls


def test_command_has_no_tools_and_a_schema(fake_claude):
    got = ClaudeCodeGuru(model="sonnet").propose("rm -rf build/", FIELDS, ["Not allowed: x"])
    assert got == {"action": "ask", "destructive": True}
    cmd = fake_claude[0]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-session-persistence" in cmd and "--strict-mcp-config" in cmd
    schema = json.loads(cmd[cmd.index("--json-schema") + 1])
    assert schema["properties"]["action"]["enum"] == ["run", "ask", "block"]
    assert schema["properties"]["destructive"] == {"type": "boolean"}
    assert "Not allowed: x" in fake_claude[0]["input"]


def test_error_result_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=json.dumps({"is_error": True, "result": "limit reached"}), stderr=""))
    with pytest.raises(RuntimeError, match="limit reached"):
        ClaudeCodeGuru().propose("x", FIELDS, [])


def test_missing_cli_raises(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda c: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        ClaudeCodeGuru().propose("x", FIELDS, [])


def test_guru_answers_are_not_trainable(fake_claude):
    schema = {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["run", "ask", "block"]}, "destructive": {"type": "boolean"}}}
    s = Shishya(FakeBackend(fixed={"action": {"run": 0.5, "ask": 0.3, "block": 0.2},
                                   "destructive": {True: 0.5, False: 0.5}}),
                guru=ClaudeCodeGuru(), min_probability=0.99)
    d = s.decide("rm -rf build/", schema)
    assert d.status == "guru" and d.answer == {"action": "ask", "destructive": True}
    assert d.trainable is False
