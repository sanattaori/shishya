"""Claude Code hook entry points for Shishya Gate.

PreToolUse asks the policy and prints its decision. Later events turn what the person did into labels:
a tool that ran after the gate had no opinion was approved, and a rejected prompt shows up in the transcript.
The hook never exits non-zero and never prints a traceback.
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PREFIX = "Shishya Gate: "
MAX_INLINE_LABELS = 1000
SCAN_AFTER_S = 2
SCAN_BYTES = 2 * 1024 * 1024
KEEP_PENDING_H = 24
REJECTED = ("doesn't want to proceed", "was rejected")


def default_dir() -> Path:
    """Where the gate learner lives unless the policy says otherwise."""
    return Path(os.environ.get("SHISHYA_GATE_DIR", "~/.shishya/gate")).expanduser()


def gate_decide(tool_name: str, tool_input: dict, cwd: str, *, home: str | None = None) -> dict:
    """The policy decision (imported when first used, so tests can replace this function)."""
    from .policy import gate_decide as decide
    return decide(tool_name, tool_input, cwd, home=home)


def _now(delta: timedelta | None = None) -> str:
    return (datetime.now(timezone.utc) + (delta or timedelta())).isoformat(timespec="seconds")


# -- pending calls ------------------------------------------------------------------------------------
@contextlib.contextmanager
def _db(learner_dir: str | os.PathLike):
    """The gate's small SQLite file (calls waiting to learn what the person did)."""
    d = Path(learner_dir).expanduser()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = d / "gate.db"
    if not path.exists():
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS pending (tool_use_id TEXT PRIMARY KEY, decision_id TEXT, gate TEXT, "
                     "permission_mode TEXT, transcript TEXT, time TEXT)")
        with conn:
            yield conn
    finally:
        conn.close()


def pending_count(learner_dir: str | os.PathLike) -> int:
    """How many calls wait for a label."""
    with _db(learner_dir) as db:
        return db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]


def _label_count(learner_dir: str | os.PathLike) -> int:
    path = Path(learner_dir).expanduser() / "learn.db"
    if not path.exists():
        return 0
    conn = sqlite3.connect(path, timeout=5)
    try:
        return conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
    finally:
        conn.close()


def _label(learner_dir: str | os.PathLike, decision_id: str, decision: str, source: str) -> None:
    """Record one label. Many labels: train in a detached process so the hook stays fast."""
    from ..learn import Learner
    big = _label_count(learner_dir) > MAX_INLINE_LABELS
    Learner(learner_dir).correct(decision_id, {"decision": decision}, source=source, train=not big)
    if big:
        subprocess.Popen([sys.executable, "-m", "shishya.cli", "learn", "train", str(Path(learner_dir).expanduser())],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)


def _safe_label(learner_dir: str | os.PathLike, decision_id: str, decision: str, source: str) -> None:
    try:
        _label(learner_dir, decision_id, decision, source)
    except Exception:  # noqa: BLE001 - a label that cannot be saved must not break the agent
        pass


# -- events -------------------------------------------------------------------------------------------
def pre_tool_use(event: dict) -> str | None:
    """Ask the policy. Returns the text to print (or None for no opinion)."""
    res = gate_decide(event.get("tool_name") or "", event.get("tool_input") or {}, event.get("cwd") or os.getcwd())
    decision = res.get("decision")
    learner_dir = res.get("learner_dir") or default_dir()
    if res.get("decision_id") and event.get("tool_use_id"):
        with _db(learner_dir) as db:
            db.execute("INSERT OR REPLACE INTO pending VALUES (?, ?, ?, ?, ?, ?)",
                       (event["tool_use_id"], res["decision_id"], decision or "none",
                        event.get("permission_mode") or "", event.get("transcript_path") or "", _now()))
    scan_rejections(event, learner_dir)
    if decision is None:
        return None
    if decision not in ("allow", "ask", "deny"):
        raise ValueError(f"the policy gave an unknown decision {decision!r}")
    return _pre_output(decision, str(res.get("reason") or ""))


def _pre_output(decision: str, reason: str) -> str:
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                              "permissionDecisionReason": PREFIX + reason}})


def post_tool_use(event: dict, learner_dir: str | os.PathLike | None = None) -> None:
    """The tool ran: if the gate had no opinion, the person approved it."""
    learner_dir = learner_dir or default_dir()
    with _db(learner_dir) as db:
        row = db.execute("SELECT decision_id, gate FROM pending WHERE tool_use_id = ?",
                         (event.get("tool_use_id"),)).fetchone()
        if row is not None:
            db.execute("DELETE FROM pending WHERE tool_use_id = ?", (event.get("tool_use_id"),))
    if row is not None and row[1] == "none":
        source = "approval" if event.get("permission_mode") == "default" else "observed"
        _safe_label(learner_dir, row[0], "allow", source)
    scan_rejections(event, learner_dir)


def _text_of(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


def _rejected_ids(transcript: str, wanted: set[str]) -> set[str]:
    """Ids among ``wanted`` whose tool_result in the last 2 MB of the transcript says the person said no."""
    found: set[str] = set()
    with open(transcript, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - SCAN_BYTES)
        f.seek(start)
        data = f.read()
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]                  # the first line is probably cut in half
    for raw in reversed(lines):
        try:
            msg = json.loads(raw)
            blocks = msg["message"]["content"]
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict) or b.get("type") != "tool_result" or b.get("tool_use_id") not in wanted:
                continue
            text = _text_of(b).lower()
            if any(w in text for w in REJECTED) or (b.get("is_error") is True and "rejected" in text):
                found.add(b["tool_use_id"])
    return found


def scan_rejections(event: dict, learner_dir: str | os.PathLike | None = None) -> None:
    """Find calls the person rejected (Claude Code sends no event for that) and learn that they needed asking."""
    try:
        learner_dir = learner_dir or default_dir()
        transcript = event.get("transcript_path")
        with _db(learner_dir) as db:
            db.execute("DELETE FROM pending WHERE time < ?", (_now(-timedelta(hours=KEEP_PENDING_H)),))
            if not transcript:
                return
            rows = db.execute("SELECT tool_use_id, decision_id, gate FROM pending WHERE transcript = ? AND time < ?",
                              (transcript, _now(-timedelta(seconds=SCAN_AFTER_S)))).fetchall()
        if not rows:
            return
        rejected = _rejected_ids(transcript, {r[0] for r in rows})
        for tool_use_id, decision_id, gate in rows:
            if tool_use_id not in rejected:
                continue
            with _db(learner_dir) as db:
                db.execute("DELETE FROM pending WHERE tool_use_id = ?", (tool_use_id,))
            if gate == "none":
                _safe_label(learner_dir, decision_id, "ask", "approval")
    except Exception:  # noqa: BLE001 - the scan is best effort
        pass


def handle(event: dict) -> str | None:
    """Dispatch one hook event by name. Returns the text to print on stdout, if any."""
    name = event.get("hook_event_name")
    if name == "PreToolUse":
        return pre_tool_use(event)
    if name == "PostToolUse":
        post_tool_use(event)
    else:
        scan_rejections(event)
    return None


def run(stdin: Any = None, stdout: Any = None) -> int:
    """Read one event from stdin, print the answer. Always returns 0."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    name = "PreToolUse"
    try:
        event = json.loads(stdin.read())
        if not isinstance(event, dict):
            raise ValueError("the event is not a JSON object")
        name = event.get("hook_event_name")
        out = handle(event)
    except Exception as e:  # noqa: BLE001 - fail closed: never a traceback, never a non-zero exit
        out = _pre_output("ask", f"could not check this call ({type(e).__name__}: {e}), so it asks") \
            if name == "PreToolUse" else None
    if out:
        print(out, file=stdout)
    return 0


def main() -> None:
    """Console entry point ``shishya-gate-hook``: the fast path Claude Code runs on every tool call."""
    sys.exit(run())
