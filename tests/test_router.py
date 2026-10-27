"""RouterBackend: each field goes to its own backend."""
from shishya import Shishya, rule
from shishya.backends import FakeBackend, RouterBackend

SCHEMA = {"type": "object", "properties": {
    "news_impact": {"type": "string", "enum": ["bad", "good", "routine"]},
    "action": {"type": "string", "enum": ["hold", "trim", "add"]}}}


class Spy(FakeBackend):
    def __init__(self, fixed, name):
        super().__init__(fixed=fixed)
        self.name, self.seen = name, []

    def score(self, text, fields, options):
        self.seen.append([f.name for f in fields])
        return super().score(text, fields, options)


def test_router_sends_each_field_to_its_backend_once():
    a = Spy({"action": {"hold": 0.6, "trim": 0.3, "add": 0.1}}, "general")
    b = Spy({"news_impact": {"bad": 0.8, "good": 0.1, "routine": 0.1}}, "specialist")
    r = RouterBackend(default=a, routes={"news_impact": b})
    no_hold_on_bad = rule(lambda ans, ctx: not (ans["news_impact"] == "bad" and ans["action"] == "hold"),
                          uses=["news_impact", "action"])
    d = Shishya(r, rules=[no_hold_on_bad]).decide("Auditor resigns", SCHEMA)
    assert a.seen == [["action"]] and b.seen == [["news_impact"]]
    assert d.answer == {"news_impact": "bad", "action": "trim"}       # the rule couples the two models
    assert "specialist" in d.receipt["backend"]["model"] or "news_impact" in d.receipt["backend"]["model"]
