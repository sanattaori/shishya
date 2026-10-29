"""OPA/Rego rules. Tests that call `opa` are skipped when the binary is not on PATH; the rest run anywhere."""
import shutil

import pytest

from shishya import Shishya
from shishya.backends import FakeBackend
from shishya.rules import RegoRules, RuleError

HAS_OPA = shutil.which("opa") is not None
needs_opa = pytest.mark.skipif(not HAS_OPA, reason="the 'opa' binary is not on PATH")

POLICY = """package shell_guard
import rego.v1

default allow := false

allow if input.answer.action != "run"

allow if {
    input.answer.action == "run"
    not input.answer.destructive
}

why_not contains "destructive command" if {
    input.answer.action == "run"
    input.answer.destructive
}
"""

SHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run", "ask"]},
        "destructive": {"type": "boolean"},
    },
}


# -- no opa needed ------------------------------------------------------------------------------------

def test_policy_without_package_line_is_rejected():
    with pytest.raises(RuleError, match="package"):
        RegoRules("default allow := false\nallow if true\n")


def test_policy_must_be_a_string():
    with pytest.raises(RuleError):
        RegoRules({"allow": True})


def test_query_defaults_to_package_allow():
    r = RegoRules(POLICY)
    assert r.package == "shell_guard"
    assert r.query == "data.shell_guard.allow"
    assert r.reasons_ref == "data.shell_guard.why_not"
    assert r.name == "rego:shell_guard"


def test_custom_query_is_kept():
    r = RegoRules(POLICY, query="data.shell_guard.ok")
    assert r.query == "data.shell_guard.ok"


def test_query_must_start_with_data():
    with pytest.raises(RuleError, match="data"):
        RegoRules(POLICY, query="shell_guard.allow")


def test_fingerprint_changes_with_policy():
    assert RegoRules(POLICY).fingerprint() != RegoRules(POLICY + "\n# changed\n").fingerprint()


def test_package_word_inside_a_comment_is_not_a_package_line():
    with pytest.raises(RuleError):
        RegoRules("# this is a package of rules\ndefault allow := false\n")


def test_check_batch_without_opa_says_so():
    r = RegoRules(POLICY)
    r.opa = None
    with pytest.raises(RuleError, match="opa"):
        r.check_batch([{"action": "ask"}], {"input": None, "facts": None})


# -- needs opa ----------------------------------------------------------------------------------------

@needs_opa
def test_check_batch_allow_flags_and_reasons():
    answers = [
        {"action": "ask", "destructive": True},
        {"action": "run", "destructive": False},
        {"action": "run", "destructive": True},
        {"action": "ask"},
    ]
    got = RegoRules(POLICY).check_batch(answers, {"input": {"text": "rm -rf"}, "facts": {}})
    assert [allowed for allowed, _ in got] == [True, True, False, True]
    assert got[2] == (False, ["destructive command"])
    assert got[0] == (True, [])
    assert got[1] == (True, [])
    assert got[3] == (True, [])


@needs_opa
def test_check_batch_empty_list_needs_no_opa():
    assert RegoRules(POLICY).check_batch([], {}) == []


@needs_opa
def test_engine_takes_next_best_when_rego_rejects_top():
    # top answer: run + destructive (0.56) -> rejected; next best: ask + destructive (0.24) -> allowed
    backend = FakeBackend(fixed={
        "action": {"run": 0.7, "ask": 0.3},
        "destructive": {True: 0.8, False: 0.2},
    })
    d = Shishya(backend, rego=POLICY).decide("delete the temp dir", SHELL_SCHEMA)

    assert d.status == "ok"
    assert d.answer == {"action": "ask", "destructive": True}
    assert d.top == {"action": "run", "destructive": True}
    assert d.top_allowed is False
    assert d.broken_rules.get("rego:shell_guard") == 1
    assert d.rejected_mass == pytest.approx(0.56)
    assert "destructive command" in d.reasons
    assert any(r["name"] == "rego:shell_guard" for r in d.receipt["rules"])


def test_failing_policy_blocks_and_is_reported():
    """opa missing, a timeout or a broken policy must block, never crash the decision or allow."""
    r = RegoRules(POLICY, opa="/nonexistent/opa")
    s = Shishya(FakeBackend(fixed={"action": {"run": 0.9, "ask": 0.1}, "destructive": {True: 0.5, False: 0.5}}),
                rego=r)
    d = s.decide("rm -rf /", SHELL_SCHEMA)
    assert d.status == "none" and d.answer is None
    assert d.rule_errors and d.rule_errors[0].startswith("rego:shell_guard:")
