"""Leave rules: the AI can approve only what the policy allows; anything else goes to a person."""
from shishya import rule


@rule(uses=["verdict"])
def within_balance(ans, ctx):
    "Leave can be approved only up to the remaining balance."
    f = ctx["facts"]
    return ans["verdict"] != "approve" or f["days_asked"] <= f["leave_balance"]


@rule(uses=["verdict"])
def probation_limit(ans, ctx):
    "People on probation can be approved for at most 2 days without HR."
    f = ctx["facts"]
    return not (ans["verdict"] == "approve" and f.get("on_probation") and f["days_asked"] > 2)


@rule(uses=["verdict", "urgent"])
def emergencies_are_never_rejected(ans, ctx):
    "An emergency is never rejected automatically."
    return not (ans["verdict"] == "reject" and ans["urgent"])
