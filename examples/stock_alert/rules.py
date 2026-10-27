"""Rules for a portfolio bot: the AI may make the portfolio safer on its own, never riskier."""
from shishya import rule


@rule(uses=["action"])
def no_buying_when_market_weak(ans, ctx):
    "Never buy more while the market guard is on (index below its 200-day average)."
    return not (ans["action"] == "add" and ctx["facts"]["market_guard_on"])


@rule(uses=["action"])
def position_size_cap(ans, ctx):
    "Never buy more of a stock that is already 10% or more of the portfolio."
    return not (ans["action"] == "add" and ctx["facts"]["weight_pct"] >= 10)


@rule(uses=["action", "tell_owner"])
def owner_hears_about_sells(ans, ctx):
    "Any sell must be told to the owner."
    return ans["action"] != "trim" or ans["tell_owner"]
