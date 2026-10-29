import io
import json
import sqlite3
from datetime import timedelta

import pytest

from shishya import Learner, Shishya
from shishya.backends import FakeBackend
from shishya.cli import main
from shishya.gate import hook

SCHEMA = {"type": "object", "properties": {"decision": {"enum": ["allow", "ask", "deny"]}}}


def fake_policy(monkeypatch, tmp_path, decision, decision_id=None):
    def decide(tool_name, tool_input, cwd, *, home=None):
        return {"decision": decision, "reason": "because", "decision_id": decision_id,
                "learner_dir": str(tmp_path), "forced": False}
    monkeypatch.setattr(hook, "gate_decide", decide)


def logged_decision(tmp_path, text="git status"):
    s = Shishya(FakeBackend(fixed={"decision": {"allow": 0.6, "ask": 0.3, "deny": 0.1}}), learner=tmp_path)
    return s.decide(text, SCHEMA).receipt["id"]


def event(name="PreToolUse", **kw):
    return {"session_id": "s", "transcript_path": "", "cwd": "/tmp", "permission_mode": "default",
            "hook_event_name": name, "tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_use_id": "t1", **kw}


def run_hook(ev):
    out = io.StringIO()
    code = hook.run(io.StringIO(json.dumps(ev)), out)
    return code, out.getvalue()


def labels(tmp_path):
    db = sqlite3.connect(tmp_path / "learn.db")
    return db.execute("SELECT field, label, source FROM labels").fetchall()


def pending(tmp_path):
    return sqlite3.connect(tmp_path / "gate.db").execute("SELECT tool_use_id FROM pending").fetchall()


@pytest.mark.parametrize("decision", ["allow", "ask", "deny"])
def test_decision_is_printed(monkeypatch, tmp_path, decision):
    fake_policy(monkeypatch, tmp_path, decision)
    code, out = run_hook(event())
    assert code == 0
    body = json.loads(out)["hookSpecificOutput"]
    assert body == {"hookEventName": "PreToolUse", "permissionDecision": decision,
                    "permissionDecisionReason": "Shishya Gate: because"}


def test_no_opinion_prints_nothing(monkeypatch, tmp_path):
    fake_policy(monkeypatch, tmp_path, None)
    assert run_hook(event()) == (0, "")


def test_exception_asks(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr(hook, "gate_decide", boom)
    code, out = run_hook(event())
    body = json.loads(out)["hookSpecificOutput"]
    assert code == 0 and body["permissionDecision"] == "ask"
    assert body["permissionDecisionReason"] == (
        "Shishya Gate: could not check this call (RuntimeError: kaput), so it asks")


def test_bad_stdin_asks():
    out = io.StringIO()
    assert hook.run(io.StringIO("not json"), out) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_other_event_error_is_silent(monkeypatch):
    monkeypatch.setattr(hook, "scan_rejections", lambda *a, **k: 1 / 0)
    assert run_hook(event("Stop")) == (0, "")


def test_approval_in_default_mode(monkeypatch, tmp_path):
    did = logged_decision(tmp_path)
    fake_policy(monkeypatch, tmp_path, None, did)
    assert run_hook(event()) == (0, "")
    assert pending(tmp_path) == [("t1",)]
    monkeypatch.setenv("SHISHYA_GATE_DIR", str(tmp_path))
    assert run_hook(event("PostToolUse")) == (0, "")
    assert labels(tmp_path) == [("decision", '"allow"', "approval")]
    assert pending(tmp_path) == []


def test_other_mode_is_observed(monkeypatch, tmp_path):
    did = logged_decision(tmp_path)
    fake_policy(monkeypatch, tmp_path, None, did)
    run_hook(event(permission_mode="acceptEdits"))
    monkeypatch.setenv("SHISHYA_GATE_DIR", str(tmp_path))
    run_hook(event("PostToolUse", permission_mode="acceptEdits"))
    assert labels(tmp_path) == [("decision", '"allow"', "observed")]


def test_own_decision_is_not_a_label(monkeypatch, tmp_path):
    did = logged_decision(tmp_path)
    fake_policy(monkeypatch, tmp_path, "allow", did)
    run_hook(event())
    assert pending(tmp_path) == [("t1",)]
    monkeypatch.setenv("SHISHYA_GATE_DIR", str(tmp_path))
    run_hook(event("PostToolUse"))
    assert labels(tmp_path) == [] and pending(tmp_path) == []


def result_line(tool_use_id, text, **kw):
    return json.dumps({"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": text, **kw}]}})


def test_rejection_in_transcript(monkeypatch, tmp_path):
    did = logged_decision(tmp_path)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("garbage\n[1, 2]\n{\"message\": 5}\n" + result_line("other", "ok") + "\n"
                          + result_line("t1", "The user doesn't want to proceed with this tool use.") + "\n{broken")
    fake_policy(monkeypatch, tmp_path, None, did)
    run_hook(event(transcript_path=str(transcript)))
    assert labels(tmp_path) == [] and pending(tmp_path) == [("t1",)]      # too fresh to scan
    db = sqlite3.connect(tmp_path / "gate.db")
    db.execute("UPDATE pending SET time = ?", (hook._now(timedelta(seconds=-10)),))
    db.commit()
    db.close()
    monkeypatch.setenv("SHISHYA_GATE_DIR", str(tmp_path))
    assert run_hook(event("UserPromptSubmit", transcript_path=str(transcript))) == (0, "")
    assert labels(tmp_path) == [("decision", '"ask"', "approval")]
    assert pending(tmp_path) == []


def test_rejection_is_error_block(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(result_line("t1", [{"type": "text", "text": "Tool use was Rejected"}], is_error=True)
                          + "\n" + result_line("t2", "all fine", is_error=False) + "\n")
    assert hook._rejected_ids(str(transcript), {"t1", "t2"}) == {"t1"}


def test_old_pending_rows_are_deleted(monkeypatch, tmp_path):
    with hook._db(tmp_path) as db:
        db.execute("INSERT INTO pending VALUES ('old', 'x', 'none', 'default', '', '2000-01-01T00:00:00+00:00')")
    hook.scan_rejections(event("Stop"), tmp_path)
    assert pending(tmp_path) == []


def test_many_labels_train_in_background(monkeypatch, tmp_path):
    did = logged_decision(tmp_path)
    started = []
    monkeypatch.setattr(hook, "MAX_INLINE_LABELS", -1)
    monkeypatch.setattr(hook.subprocess, "Popen", lambda cmd, **kw: started.append((cmd, kw)))
    hook._label(tmp_path, did, "allow", "approval")
    assert started and started[0][0][-3:] == ["learn", "train", str(tmp_path)]
    assert started[0][1]["start_new_session"] is True
    assert Learner(tmp_path).status()["fields"]["decision"]["train"] + \
        Learner(tmp_path).status()["fields"]["decision"]["test"] == 1


def test_gate_test_command(monkeypatch, tmp_path, capsys):
    seen = {}

    def decide(tool_name, tool_input, cwd, *, home=None):
        seen.update(tool=tool_name, input=tool_input)
        return {"decision": "deny", "reason": "no force pushes", "decision_id": None,
                "learner_dir": str(tmp_path), "forced": True}
    monkeypatch.setattr(hook, "gate_decide", decide)
    assert main(["gate", "test", "git push --force"]) == 0
    out = capsys.readouterr().out
    assert "deny" in out and "no force pushes" in out
    assert seen == {"tool": "Bash", "input": {"command": "git push --force"}}
    assert main(["gate", "test", "a.txt", "--tool", "Write", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["forced"] is True and seen["input"] == {"file_path": "a.txt"}


def test_gate_status_and_correct(monkeypatch, tmp_path, capsys):
    did = logged_decision(tmp_path)
    monkeypatch.setenv("SHISHYA_GATE_DIR", str(tmp_path))
    assert main(["gate", "correct", did, "ask"]) == 0
    assert labels(tmp_path) == [("decision", '"ask"', "human")]
    assert main(["gate", "status"]) == 0
    assert "0 calls waiting" in capsys.readouterr().out
