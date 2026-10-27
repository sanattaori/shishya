"""The search must return exactly what brute force returns: the most likely answer that passes."""
import itertools
import math
import random

import pytest

from shishya.rules import rule
from shishya.search import search


def brute_force(order, probs, rules, ctx, fixed, checker=None):
    best = None
    for combo in itertools.product(*[list(probs[n]) for n in order]):
        ans = dict(fixed)
        ans.update(zip(order, combo))
        if not all(r(ans, ctx) for r in rules):
            continue
        if checker is not None and not checker(ans):
            continue
        lp = sum(math.log(max(probs[n][v], 1e-12)) for n, v in zip(order, combo))
        if best is None or lp > best[1]:
            best = (ans, lp)
    return best


def random_problem(rng):
    n = rng.randint(1, 4)
    order = [f"f{i}" for i in range(n)]
    probs = {}
    for name in order:
        k = rng.randint(2, 5)
        w = [rng.random() + 1e-3 for _ in range(k)]
        probs[name] = {f"o{j}": x / sum(w) for j, x in enumerate(w)}
    rules = []
    for _ in range(rng.randint(0, 3)):
        uses = rng.sample(order, rng.randint(1, n)) if rng.random() < 0.7 else None
        bad = {(name, f"o{rng.randint(0, 4)}") for name in order if rng.random() < 0.5}

        def fn(ans, ctx, bad=frozenset(bad), uses=tuple(uses) if uses else tuple(order)):
            return not any(ans.get(name) == v for name, v in bad if name in uses)
        rules.append(rule(fn, uses=uses))
    return order, probs, rules


class SetChecker:
    """A batch checker (like Rego) that forbids some complete answers."""
    name = "set"

    def __init__(self, forbidden):
        self.forbidden = forbidden
        self.calls = 0

    def __call__(self, ans):
        return tuple(sorted(ans.items())) not in self.forbidden

    def check_batch(self, answers, ctx):
        self.calls += 1
        return [(self(a), ["forbidden"] if not self(a) else []) for a in answers]


@pytest.mark.parametrize("seed", range(300))
def test_matches_brute_force(seed):
    rng = random.Random(seed)
    order, probs, rules = random_problem(rng)
    checker = None
    if rng.random() < 0.4:
        all_answers = [tuple(sorted(zip(order, c))) for c in itertools.product(*[list(probs[n]) for n in order])]
        checker = SetChecker(set(rng.sample(all_answers, rng.randint(0, len(all_answers)))))
    got = search(order, probs, rules=rules, checkers=[checker] if checker else [], ctx={},
                 max_candidates=10_000, batch_size=rng.randint(1, 8))
    want = brute_force(order, probs, rules, {}, {}, checker)
    if want is None:
        assert got.status == "impossible" and got.answer is None
    else:
        assert got.status == "ok"
        assert got.answer == want[0]
        assert math.isclose(got.logp, want[1], rel_tol=1e-9)
        assert 0.0 <= got.rejected_mass <= 1.0 + 1e-9


def test_top_answer_passes_first_try():
    probs = {"a": {"x": 0.9, "y": 0.1}, "b": {True: 0.8, False: 0.2}}
    got = search(["a", "b"], probs)
    assert got.answer == {"a": "x", "b": True}
    assert got.tried == 1 and got.rejected_mass == 0


def test_rule_moves_to_next_best_and_prunes_branch():
    probs = {"a": {"x": 0.9, "y": 0.1}, "b": {"p": 0.6, "q": 0.4}}
    no_x = rule(lambda ans, ctx: ans["a"] != "x", uses=["a"], name="no_x")
    got = search(["a", "b"], probs, rules=[no_x])
    assert got.answer == {"a": "y", "b": "p"}
    assert got.failures["no_x"] == 1            # the whole a=x branch went in one check
    assert math.isclose(got.rejected_mass, 0.9)


def test_fixed_fields_are_visible_to_rules():
    probs = {"a": {"x": 0.7, "y": 0.3}}
    r = rule(lambda ans, ctx: not (ans["mode"] == "strict" and ans["a"] == "x"), uses=["a", "mode"])
    got = search(["a"], probs, rules=[r], fixed={"mode": "strict"})
    assert got.answer == {"mode": "strict", "a": "y"}


def test_rule_on_fixed_only_can_make_it_impossible():
    r = rule(lambda ans, ctx: ans["mode"] != "off", uses=["mode"])
    got = search(["a"], {"a": {"x": 1.0}}, rules=[r], fixed={"mode": "off"})
    assert got.status == "impossible"


def test_candidate_budget_reports_exhausted():
    probs = {f"f{i}": {"a": 0.5, "b": 0.5} for i in range(6)}
    checker = SetChecker(set())
    checker.check_batch = lambda answers, ctx: [(False, []) for _ in answers]   # rejects everything
    got = search(list(probs), probs, checkers=[checker], max_candidates=10, batch_size=4)
    assert got.status == "exhausted" and got.answer is None and got.tried >= 10


def test_unknown_rule_field_is_an_error():
    with pytest.raises(ValueError):
        search(["a"], {"a": {"x": 1.0}}, rules=[rule(lambda a, c: True, uses=["nope"])])


def test_ctx_reaches_rules():
    r = rule(lambda ans, ctx: ans["a"] == ctx["facts"]["want"], uses=["a"])
    got = search(["a"], {"a": {"x": 0.9, "y": 0.1}}, rules=[r], ctx={"facts": {"want": "y"}})
    assert got.answer == {"a": "y"}
