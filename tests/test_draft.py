"""English -> rules drafter: a fake Guru drafts, every rule is validated, nothing is applied."""
from __future__ import annotations

import json
import subprocess

import pytest

from shishya.cli import main
from shishya.draft import DraftError, draft_rules, examples, system_prompt
from shishya.guru import ClaudeCodeGuru

SCHEMA = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["hold", "watch", "trim", "add"], "description": "What to do?"},
    "tell_owner": {"type": "boolean"}}}
FACTS = {"market_guard_on": True, "cash": 1200}
GOOD = {"name": "no_buying_when_market_weak", "message": "Never buy while the market guard is on.",
        "forbid": {"all": [{"path": "answer.action", "eq": "add"}, {"path": "facts.market_guard_on", "eq": True}]}}
BAD = {"name": "oops", "message": "m", "forbid": {"path": "answer.nope", "eq": 1}}


class FakeGuru:
    name = "fake"
    model = "fake-0"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, system, prompt, json_schema):
        self.calls.append((system, prompt, json_schema))
        return self.replies.pop(0)


def test_valid_draft_and_examples():
    guru = FakeGuru({"rules": [GOOD]})
    rules = draft_rules(guru, SCHEMA, "Never buy while the market guard is on.", FACTS)
    assert rules == [GOOD] and len(guru.calls) == 1
    assert guru.calls[0][2]["properties"]["rules"]["type"] == "array"
    rows = examples(SCHEMA, rules, FACTS)
    blocked = [r for r in rows if r["blocked_by"]]
    allowed = [r for r in rows if not r["blocked_by"]]
    assert blocked and allowed
    assert all(r["answer"]["action"] == "add" and r["blocked_by"] == [GOOD["name"]] for r in blocked)
    # the allowed row shown is the closest one to the blocked one
    assert allowed[0]["answer"]["tell_owner"] == blocked[0]["answer"]["tell_owner"]
    # without the fact the rule never fires
    # rules fail closed: with no facts at all, the rule's fact is missing, so "add" is blocked
    assert any(r["blocked_by"] for r in examples(SCHEMA, rules, {}))


def test_invalid_draft_retries_once_with_the_errors():
    guru = FakeGuru({"rules": [BAD]}, {"rules": [GOOD]})
    assert draft_rules(guru, SCHEMA, "x", FACTS) == [GOOD]
    assert len(guru.calls) == 2
    retry = guru.calls[1][1]
    assert "no such field" in retry and "answer.nope" in retry


def test_still_invalid_raises_with_all_problems():
    guru = FakeGuru({"rules": [BAD]}, {"rules": [BAD, {"name": "Bad Name", "forbid": {"path": "answer.action",
                                                                                      "eq": "buy"}}]})
    with pytest.raises(DraftError) as e:
        draft_rules(guru, SCHEMA, "x", FACTS)
    msg = str(e.value)
    assert "no such field" in msg and "snake_case" in msg and "can never be 'buy'" in msg
    assert len(guru.calls) == 2


def test_reply_of_the_wrong_shape_is_a_problem():
    with pytest.raises(DraftError, match="rules"):
        draft_rules(FakeGuru([1], "nope"), SCHEMA, "x")


def test_system_prompt_lists_fields_options_and_facts():
    p = system_prompt(SCHEMA, FACTS)
    for needle in ("answer.action", "answer.tell_owner", "hold", "watch", "trim", "add", "true", "false",
                   "facts.market_guard_on", "facts.cash", "1200", "forbid", "require", "snake_case",
                   "eq", "neq", "gt", "gte", "lt", "lte", "contains", "exists", "all", "any", "not"):
        assert needle in p
    assert "fail closed" in p


def test_examples_cap_and_limit():
    wide = {"type": "object", "properties": {f"f{i}": {"type": "boolean"} for i in range(14)}}
    rule = {"name": "r", "forbid": {"path": "answer.f0", "eq": True}}
    rows = examples(wide, [rule], {}, limit=5)
    assert 0 < len(rows) <= 5 and any(r["blocked_by"] for r in rows)


def test_cli_rules_end_to_end(tmp_path, capsys, monkeypatch):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "f.json").write_text(json.dumps(FACTS))
    guru = FakeGuru({"rules": [GOOD]})
    monkeypatch.setattr("shishya.cli._guru", lambda name: guru)
    out = tmp_path / "rules.json"
    code = main(["rules", str(tmp_path / "s.json"), "Never buy while the market guard is on.",
                 "--facts", str(tmp_path / "f.json"), "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "no_buying_when_market_weak" in text and "BLOCK" in text and "allow" in text
    assert text.rstrip().endswith("Review these rules, then add them to the schema under x-shishya.rules. "
                                  "Nothing was applied.")
    assert json.loads(out.read_text()) == {"rules": [GOOD]}
    assert "rules" not in json.loads((tmp_path / "s.json").read_text()).get("x-shishya", {})


def test_cli_rules_reports_failure(tmp_path, capsys, monkeypatch):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    monkeypatch.setattr("shishya.cli._guru", lambda name: FakeGuru({"rules": [BAD]}, {"rules": [BAD]}))
    assert main(["rules", str(tmp_path / "s.json"), "x"]) == 1
    assert "not valid" in capsys.readouterr().err


def test_claude_complete_json_command(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append({"cmd": cmd, **kw})
        body = {"is_error": False, "structured_output": {"rules": []}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(body), stderr="")
    monkeypatch.setattr("shutil.which", lambda c: "/usr/bin/claude")
    monkeypatch.setattr("subprocess.run", run)
    schema = {"type": "object", "properties": {"rules": {"type": "array"}}}
    assert ClaudeCodeGuru().complete_json("SYS", "the prompt", schema) == {"rules": []}
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema
    assert calls[0]["input"] == "the prompt"


def test_invented_fact_keys_are_rejected_when_facts_are_described():
    from shishya.draft import _validate
    schema = {"type": "object", "properties": {"action": {"type": "string", "enum": ["hold", "add"]}}}
    rule = {"name": "no_add", "message": "m", "forbid": {"all": [{"path": "answer.action", "eq": "add"},
                                                                 {"path": "facts.made_up", "eq": True}]}}
    good, problems = _validate({"rules": [rule]}, schema, {"market_guard_on": True})
    assert not good and "facts.made_up" in problems[0]
    good, problems = _validate({"rules": [rule]}, schema, None)       # no facts described: nothing to check
    assert good and not problems
