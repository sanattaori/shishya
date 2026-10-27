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


@rule(uses=["news_impact", "action"])
def act_with_the_news(ans, ctx):
    "Trade only with the news: no buying on bad news, no selling on good news, no trades on neutral news."
    impact, action = ans["news_impact"], ans["action"]
    if impact == "bearish":
        return action in ("watch", "trim")
    if impact == "bullish":
        return action in ("hold", "watch", "add")
    return action in ("hold", "watch")
