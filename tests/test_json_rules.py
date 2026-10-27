"""Declarative JSON rules in x-shishya.rules: safe, prune early, and agree with the equivalent Python rule."""
import pytest

from shishya import Shishya, rule
from shishya.backends import FakeBackend
from shishya.rules import RuleError, json_rule
from shishya.schema import SchemaError, compile_schema

PROPS = {"action": {"type": "string", "enum": ["hold", "watch", "trim", "add"]},
         "tell_owner": {"type": "boolean"}}
PROBS = {"action": {"add": 0.5, "hold": 0.3, "watch": 0.15, "trim": 0.05}, "tell_owner": {True: 0.4, False: 0.6}}
GUARD = {"name": "no_buying_when_market_weak", "message": "Never buy while the market guard is on.",
         "forbid": {"all": [{"path": "answer.action", "eq": "add"}, {"path": "facts.market_guard_on", "eq": True}]}}


def schema(*rules):
    return {"type": "object", "properties": PROPS, "x-shishya": {"rules": list(rules)}}


def test_forbid_rule_moves_to_next_best():
    d = Shishya(FakeBackend(fixed=PROBS)).decide("good news", schema(GUARD), facts={"market_guard_on": True})
    assert d.answer == {"action": "hold", "tell_owner": False}
    assert d.broken_rules == {"no_buying_when_market_weak": 1}


def test_forbid_rule_is_quiet_only_when_the_fact_is_false():
    s = Shishya(FakeBackend(fixed=PROBS))
    assert s.decide("x", schema(GUARD), facts={"market_guard_on": False}).answer["action"] == "add"


def test_missing_fact_blocks_and_is_reported():
    d = Shishya(FakeBackend(fixed=PROBS)).decide("x", schema(GUARD), facts={})
    assert d.answer["action"] == "hold"                       # fail closed: "add" is not allowed without the fact
    assert d.missing_facts == ["facts.market_guard_on"]
    assert any("not sent" in r for r in d.reasons)


def test_on_missing_allow_opts_out():
    lenient = {**GUARD, "on_missing": "allow"}
    d = Shishya(FakeBackend(fixed=PROBS)).decide("x", schema(lenient), facts={})
    assert d.answer["action"] == "add" and d.missing_facts == ["facts.market_guard_on"]
    with pytest.raises(RuleError, match="cannot allow"):
        json_rule({"name": "r", "require": {"path": "answer.action", "eq": "add"}, "on_missing": "allow"},
                  {"action"})


def test_python_rule_that_crashes_blocks_instead_of_crashing():
    @rule(uses=["action"])
    def needs_guard(ans, ctx):
        return not (ans["action"] == "add" and ctx["facts"]["market_guard_on"])
    plain = {"type": "object", "properties": PROPS}
    d = Shishya(FakeBackend(fixed=PROBS), rules=[needs_guard]).decide("x", plain, facts={})
    assert d.status == "ok" and d.answer["action"] == "hold"
    assert d.rule_errors and "KeyError" in d.rule_errors[0]


def test_require_rule_fails_on_unknown_facts():
    req = {"name": "sells_are_told", "require": {"any": [{"path": "answer.action", "neq": "trim"},
                                                          {"path": "answer.tell_owner", "eq": True}]}}
    probs = {"action": {"trim": 0.9, "hold": 0.1}, "tell_owner": {True: 0.2, False: 0.8}}
    d = Shishya(FakeBackend(fixed=probs)).decide("x", {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["trim", "hold"]}, "tell_owner": {"type": "boolean"}},
        "x-shishya": {"rules": [req]}})
    assert d.answer == {"action": "trim", "tell_owner": True}


def test_json_rule_matches_python_rule():
    @rule(uses=["action"])
    def py_guard(ans, ctx):
        return not (ans["action"] == "add" and ctx["facts"]["market_guard_on"])
    plain = {"type": "object", "properties": PROPS}
    for guard in (True, False):
        a = Shishya(FakeBackend(fixed=PROBS)).decide("x", schema(GUARD), facts={"market_guard_on": guard})
        b = Shishya(FakeBackend(fixed=PROBS), rules=[py_guard]).decide("x", plain, facts={"market_guard_on": guard})
        assert a.answer == b.answer and abs(a.probability - b.probability) < 1e-12


def test_rule_uses_only_answer_fields_it_reads():
    r = json_rule(GUARD, {"action", "tell_owner"})
    assert r.uses == ("action",) and r.message.startswith("Never buy")
    assert r.fingerprint() == json_rule(dict(GUARD), {"action", "tell_owner"}).fingerprint()


@pytest.mark.parametrize("bad, msg", [
    ({"name": "x", "forbid": {"path": "answer.nope", "eq": 1}}, "no such field"),
    ({"name": "x", "forbid": {"path": "facts.a", "eq": 1}}, "never reads the answer"),
    ({"name": "x"}, "exactly one"),
    ({"name": "x", "forbid": {"path": "answer.action", "eq": "add"}, "require": {"exists": "facts.a"}}, "exactly one"),
    ({"forbid": {"path": "answer.action", "eq": "add"}}, "needs a 'name'"),
    ({"name": "x", "forbid": {"path": "answer.action", "bogus": 1}}, "operator"),
    ({"name": "x", "forbid": {"path": "answer.action", "eq": "add"}, "code": "import os"}, "unknown keys"),
])
def test_bad_json_rules_are_rejected(bad, msg):
    with pytest.raises(RuleError, match=msg):
        json_rule(bad, {"action", "tell_owner"})


def test_schema_level_errors():
    with pytest.raises(SchemaError, match="unique"):
        compile_schema(schema(GUARD, GUARD))
    with pytest.raises(SchemaError, match="cannot read 'answer'"):
        compile_schema({"type": "object", "properties": {
            "a": {"type": "boolean", "x-shishya": {"when": {"path": "answer.b", "eq": True}}},
            "b": {"type": "boolean"}}})
    with pytest.raises(SchemaError, match="unknown keys"):
        compile_schema({"type": "object", "properties": PROPS, "x-shishya": {"rulez": []}})


def test_rules_in_schema_change_the_digest_and_show_in_receipt():
    d = Shishya(FakeBackend(fixed=PROBS)).decide("x", schema(GUARD), facts={"market_guard_on": True})
    assert any(r["name"] == "no_buying_when_market_weak" for r in d.receipt["rules"])
    assert compile_schema(schema(GUARD)).digest != compile_schema(schema()).digest
