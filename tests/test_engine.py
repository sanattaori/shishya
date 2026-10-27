"""The engine end to end, with FakeBackend(fixed=...) so probabilities are exact."""
import pytest

from shishya import Shishya
from shishya.backends import FakeBackend
from shishya.rules import rule
from shishya.schema import digest

VERDICT = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "deny", "review"], "description": "Decide the request."},
    },
}
VERDICT_PROBS = {"approve": 0.6, "deny": 0.3, "review": 0.1}


@rule(uses=["verdict"])
def no_approve(ans, ctx):
    "Approval is not allowed here."
    return ans["verdict"] != "approve"


@rule(uses=["verdict"])
def nothing_allowed(ans, ctx):
    "Nothing is allowed."
    return False


class SpyBackend(FakeBackend):
    """FakeBackend that also records which fields were sent to it."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list[list[str]] = []

    def score(self, text, fields, options):
        self.seen.append([f.name for f in fields])
        return super().score(text, fields, options)


class FakeGuru:
    name = "fake-guru"
    model = "guru-0"
    max_tries = 3

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def propose(self, text, fields, feedback):
        self.calls.append({"text": text, "fields": fields, "feedback": list(feedback)})
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


# -- rules ---------------------------------------------------------------------------------------------

def test_rule_forbidding_top_option_gives_next_best():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS})
    d = Shishya(backend, rules=[no_approve]).decide("Need 5 days off", VERDICT)

    assert d.status == "ok"
    assert d.answer == {"verdict": "deny"}
    assert d.top == {"verdict": "approve"}
    assert d.top_allowed is False
    assert d.broken_rules == {"no_approve": 1}
    assert d.rejected_mass == pytest.approx(0.6)
    assert d.probability == pytest.approx(0.3)


def test_rule_passed_per_call_is_used():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS})
    d = Shishya(backend).decide("text", VERDICT, rules=[no_approve])
    assert d.answer == {"verdict": "deny"}


def test_top_allowed_true_when_top_option_passes():
    backend = FakeBackend(fixed={"verdict": {"deny": 0.7, "approve": 0.2, "review": 0.1}})
    d = Shishya(backend, rules=[no_approve]).decide("text", VERDICT)
    assert d.answer == {"verdict": "deny"}
    assert d.top == {"verdict": "deny"}
    assert d.top_allowed is True
    # approve is still checked (and ruled out) while the search runs, even though deny is the top answer
    assert d.broken_rules == {"no_approve": 1}
    assert d.rejected_mass == pytest.approx(0.2)


def test_every_option_forbidden_is_none():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS})
    d = Shishya(backend, rules=[nothing_allowed]).decide("text", VERDICT)
    assert d.status == "none"
    assert d.answer is None
    assert d.suggestion is None
    assert d.top_allowed is False
    assert d.rejected_mass == pytest.approx(1.0)
    assert d.broken_rules == {"nothing_allowed": 3}


# -- conditions ----------------------------------------------------------------------------------------

REFUND_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "deny"]},
        "notify_hr": {"type": "boolean", "x-shishya": {"when": {"path": "facts.has_order", "eq": True}}},
    },
}


def test_field_when_false_is_none_and_not_sent_to_backend():
    backend = SpyBackend(fixed={"verdict": {"approve": 0.8, "deny": 0.2}})
    d = Shishya(backend).decide("text", REFUND_SCHEMA, facts={"has_order": False})

    assert d.status == "ok"
    assert d.answer == {"verdict": "approve", "notify_hr": None}
    assert backend.seen == [["verdict"]]
    assert d.fields["notify_hr"] == {"value": None, "probability": None, "options": None}


def test_field_when_unknown_fact_is_none():
    backend = SpyBackend(fixed={"verdict": {"approve": 0.8, "deny": 0.2}})
    d = Shishya(backend).decide("text", REFUND_SCHEMA, facts={})
    assert d.answer == {"verdict": "approve", "notify_hr": None}
    assert "notify_hr" not in backend.seen[0]


def test_option_when_removes_option():
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "ask", "block"],
                "x-shishya": {"option_when": {"run": {"path": "facts.approved", "eq": True}}},
            },
        },
    }
    # no "run" in the fixed probabilities: the option is gone, so the backend must not need it
    backend = SpyBackend(fixed={"action": {"ask": 0.6, "block": 0.4}})
    d = Shishya(backend).decide("text", schema, facts={"approved": False})

    assert d.answer == {"action": "ask"}
    assert d.fields["action"]["options"] == {"ask": 0.6, "block": 0.4}


def test_single_remaining_option_is_fixed_without_backend_for_that_field():
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "ask"],
                "x-shishya": {"option_when": {"run": {"path": "facts.approved", "eq": True}}},
            },
            "verdict": {"type": "string", "enum": ["approve", "deny"]},
        },
    }
    backend = SpyBackend(fixed={"verdict": {"approve": 0.3, "deny": 0.7}})
    d = Shishya(backend).decide("text", schema, facts={"approved": False})

    assert backend.seen == [["verdict"]]
    assert d.answer == {"action": "ask", "verdict": "deny"}
    assert d.fields["action"] == {"value": "ask", "probability": 1.0, "options": None}
    assert d.fields["verdict"]["options"] == {"approve": 0.3, "deny": 0.7}


def test_all_fields_fixed_means_no_backend_call():
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["ask"]},
            "notify_hr": {"type": "boolean", "x-shishya": {"when": {"path": "facts.has_order", "eq": True}}},
        },
    }
    backend = FakeBackend()
    d = Shishya(backend).decide("text", schema, facts={"has_order": False})

    assert backend.calls == 0
    assert d.status == "ok"
    assert d.answer == {"action": "ask", "notify_hr": None}


# -- thresholds ----------------------------------------------------------------------------------------

def test_min_probability_above_answer_is_unsure():
    backend = FakeBackend(fixed={"verdict": {"approve": 0.2, "deny": 0.5, "review": 0.3}})
    d = Shishya(backend, min_probability=0.6).decide("text", VERDICT)

    assert d.status == "unsure"
    assert d.answer is None
    assert d.suggestion == {"verdict": "deny"}
    assert d.probability == pytest.approx(0.5)


def test_min_probability_at_or_below_answer_is_ok():
    backend = FakeBackend(fixed={"verdict": {"approve": 0.2, "deny": 0.5, "review": 0.3}})
    d = Shishya(backend, min_probability=0.5).decide("text", VERDICT)
    assert d.status == "ok"
    assert d.answer == {"verdict": "deny"}
    assert d.suggestion is None


# -- guru ----------------------------------------------------------------------------------------------

def test_guru_fixes_invalid_then_valid_answer_with_feedback():
    backend = FakeBackend(fixed={"verdict": {"approve": 0.5, "deny": 0.3, "review": 0.2}})
    guru = FakeGuru([{"verdict": "maybe"}, {"verdict": "review"}])
    d = Shishya(backend, rules=[no_approve], guru=guru, min_probability=0.5).decide("text", VERDICT)

    assert d.status == "guru"
    assert d.answer == {"verdict": "review"}
    assert d.probability == 0.0
    assert len(guru.calls) == 2
    second_feedback = guru.calls[1]["feedback"]
    assert any("'verdict' must be one of" in f for f in second_feedback)
    # the first call was told about the broken rule
    assert "Not allowed: Approval is not allowed here." in guru.calls[0]["feedback"]


def test_guru_is_not_called_when_rules_give_ok():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS})
    guru = FakeGuru([{"verdict": "review"}])
    d = Shishya(backend, rules=[no_approve], guru=guru).decide("text", VERDICT)
    assert d.status == "ok"
    assert guru.calls == []


def test_guru_that_always_breaks_the_rule_is_none_with_reason():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS})
    guru = FakeGuru([{"verdict": "approve"}])
    d = Shishya(backend, rules=[nothing_allowed], guru=guru).decide("text", VERDICT)

    assert d.status == "none"
    assert d.answer is None
    assert len(guru.calls) == guru.max_tries
    assert any("guru" in r.lower() for r in d.reasons)
    assert d.receipt["guru"] == {"name": "fake-guru", "model": "guru-0"}


# -- output shape --------------------------------------------------------------------------------------

def test_answer_keys_come_back_in_property_order():
    schema = {
        "type": "object",
        "properties": {
            "zeta": {"type": "boolean"},
            "alpha": {"type": "string", "enum": ["x", "y"]},
        },
    }
    backend = FakeBackend(fixed={"alpha": {"x": 0.9, "y": 0.1}, "zeta": {True: 0.2, False: 0.8}})
    d = Shishya(backend).decide("text", schema)
    assert list(d.answer) == ["zeta", "alpha"]
    assert d.answer == {"zeta": False, "alpha": "x"}


def test_answer_keys_follow_order_key():
    schema = {
        "type": "object",
        "properties": {
            "zeta": {"type": "boolean"},
            "alpha": {"type": "string", "enum": ["x", "y"]},
        },
        "x-shishya": {"order": ["alpha", "zeta"]},
    }
    backend = FakeBackend(fixed={"alpha": {"x": 0.9, "y": 0.1}, "zeta": {True: 0.2, False: 0.8}})
    d = Shishya(backend).decide("text", schema)
    assert list(d.answer) == ["alpha", "zeta"]


def test_reordered_properties_do_not_reuse_cached_order():
    props = {
        "zeta": {"type": "boolean"},
        "alpha": {"type": "string", "enum": ["x", "y"]},
    }
    first = {"type": "object", "properties": dict(props)}
    second = {"type": "object", "properties": dict(reversed(list(props.items())))}
    backend = FakeBackend(fixed={"alpha": {"x": 0.9, "y": 0.1}, "zeta": {True: 0.2, False: 0.8}})
    s = Shishya(backend)
    s.decide("text", first)
    d = s.decide("text", second)
    assert list(d.answer) == ["alpha", "zeta"]


def test_receipt_has_schema_rules_backend_and_timings():
    backend = FakeBackend(fixed={"verdict": VERDICT_PROBS}, model="fake-7")
    d = Shishya(backend, rules=[no_approve]).decide("text", VERDICT)

    assert d.receipt["schema_digest"] == digest(VERDICT)
    assert d.receipt["schema_digest"].startswith("sha256:")
    names = [r["name"] for r in d.receipt["rules"]]
    assert names == ["no_approve"]
    fingerprints = [r["fingerprint"] for r in d.receipt["rules"]]
    assert fingerprints == [no_approve.fingerprint()]
    assert d.receipt["backend"] == {"name": "fake", "model": "fake-7"}
    assert d.receipt["timings_ms"]["total_ms"] >= 0
    assert "model_ms" in d.receipt["timings_ms"]
    assert "search_ms" in d.receipt["timings_ms"]
