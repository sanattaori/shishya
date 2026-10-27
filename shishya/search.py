"""Best-first search for the most likely answer that follows the rules.

Each field has a probability for every allowed option (from the model). Treating fields as independent,
an answer's score is the sum of its fields' log-probabilities. The search is A*: a partial answer's
priority is its score so far plus the best possible score of the fields still open. That bound is
exact, so complete answers come out in true best-first order, and the first complete answer that
passes every rule is the most likely allowed answer.

Rules with ``uses`` are checked as soon as their fields are decided; when one fails, every answer in
that branch is ruled out without being enumerated. Rules without ``uses`` and batch checkers (Rego)
see complete answers only; complete answers are collected in best-first order and checked a batch at
a time, which keeps the result exact (everything before the first allowed answer in the batch was
checked and rejected, everything after it is less likely).
"""
from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .rules import Rule

FLOOR = 1e-12      # probabilities below this count as this, so log() stays finite


class BatchChecker(Protocol):
    name: str

    def check_batch(self, answers: list[dict], ctx: dict) -> list[tuple[bool, list[str]]]: ...


@dataclass
class SearchResult:
    status: str                         # "ok" | "exhausted" (budget hit) | "impossible" (no allowed answer exists)
    answer: dict | None = None
    logp: float = float("-inf")
    tried: int = 0                      # complete answers checked
    expanded: int = 0                   # partial answers expanded
    rejected_mass: float = 0.0          # probability of all answers the rules ruled out during the search
    failures: Counter = field(default_factory=Counter)    # rule name -> times it rejected something
    reasons: list[str] = field(default_factory=list)      # policy reasons (Rego deny_reasons), deduplicated

    @property
    def probability(self) -> float:
        return math.exp(self.logp) if self.answer is not None else 0.0


def normalize(probs: dict[Any, float]) -> dict[Any, float]:
    clean = {k: max(float(v), 0.0) for k, v in probs.items()}
    total = sum(clean.values())
    if total <= 0:
        n = len(clean)
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def search(order: list[str], probs: dict[str, dict[Any, float]], *, rules: list[Rule] = (),
           checkers: list[BatchChecker] = (), ctx: dict | None = None, fixed: dict | None = None,
           max_candidates: int = 64, max_expanded: int = 100_000, batch_size: int = 32,
           on_candidate: Callable[[dict, float], None] | None = None) -> SearchResult:
    """Find the most likely complete answer over ``order`` that passes ``rules`` and ``checkers``.

    ``probs[name]`` maps each allowed option of a field to its probability (normalised here).
    ``fixed`` holds fields that are not searched (decided by conditions) and is visible to rules.
    """
    ctx = ctx or {}
    fixed = dict(fixed or {})
    n = len(order)
    res = SearchResult(status="impossible")
    names = set(order) | set(fixed)
    for r in rules:
        if r.uses is not None and not set(r.uses) <= names:
            raise ValueError(f"rule {r.name!r} uses unknown fields {sorted(set(r.uses) - names)}")

    # options per field, most likely first, as (option, logp)
    opts: list[list[tuple[Any, float]]] = []
    for name in order:
        p = normalize(probs[name])
        if not p:
            return res                  # a field with no allowed option: nothing can be answered
        opts.append(sorted(((o, math.log(max(v, FLOOR))) for o, v in p.items()), key=lambda t: -t[1]))
    best_rest = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        best_rest[i] = best_rest[i + 1] + opts[i][0][1]

    # rules checked when the field at depth d is decided: the deepest field they use
    pos = {name: i for i, name in enumerate(order)}
    at_depth: list[list[Rule]] = [[] for _ in range(max(n, 1))]
    on_complete: list[Rule] = []
    for r in rules:
        if r.uses is None:
            on_complete.append(r)
            continue
        searched = [pos[u] for u in r.uses if u in pos]
        if not searched:                # uses only fixed fields: check once, up front
            if not r(fixed, ctx):
                res.failures[r.name] += 1
                return res
            continue
        at_depth[max(searched)].append(r)

    def as_answer(choice: tuple) -> dict:
        out = dict(fixed)
        out.update(zip(order, choice))
        return out

    def passes(rs: list[Rule], out: dict) -> bool:
        for r in rs:
            if not r(out, ctx):
                res.failures[r.name] += 1
                return False
        return True

    if n == 0:                          # everything fixed: just check the complete answer
        out = dict(fixed)
        if not passes(on_complete, out):
            return res
        for chk in checkers:
            ok, why = chk.check_batch([out], ctx)[0]
            if not ok:
                res.failures[chk.name] += 1
                res.reasons.extend(w for w in why if w not in res.reasons)
                return res
        res.status, res.answer, res.logp, res.tried = "ok", out, 0.0, 1
        return res

    tie = itertools.count()
    heap: list = [(-best_rest[0], next(tie), 0.0, ())]
    pending: list[tuple[dict, float]] = []          # complete answers waiting for the batch checkers

    def flush() -> bool:
        """Check pending complete answers in order; True when one passed (sets res)."""
        outs = [o for o, _ in pending]
        verdicts: list[tuple[str | None, list[str]]] = [(None, [])] * len(outs)   # (rejecting checker, reasons)
        for chk in checkers:
            todo = [i for i, (who, _) in enumerate(verdicts) if who is None]
            if not todo:
                break
            got = chk.check_batch([outs[i] for i in todo], ctx)
            verdicts = list(verdicts)
            for i, (ok, why) in zip(todo, got):
                if not ok:
                    verdicts[i] = (chk.name, why)
        # only answers before the winner count: the rest of the batch was never "tried"
        for (out, lp), (who, why) in zip(pending, verdicts):
            res.tried += 1
            if who is None:
                res.status, res.answer, res.logp = "ok", out, lp
                pending.clear()
                return True
            res.failures[who] += 1
            res.reasons.extend(w for w in why if w not in res.reasons)
            res.rejected_mass += math.exp(lp)
        pending.clear()
        return False

    while heap:
        if res.expanded >= max_expanded:
            res.status = "exhausted"
            break
        _, _, g, choice = heapq.heappop(heap)
        depth = len(choice)
        if depth == n:
            out = as_answer(choice)
            if on_candidate:
                on_candidate(out, g)
            if not checkers:
                res.tried += 1
                res.status, res.answer, res.logp = "ok", out, g
                return res
            pending.append((out, g))
            if len(pending) >= batch_size or res.tried + len(pending) >= max_candidates or not heap:
                if flush():
                    return res
                if res.tried >= max_candidates:
                    res.status = "exhausted"
                    return res
            continue
        res.expanded += 1
        for opt, lp in opts[depth]:
            child = choice + (opt,)
            g2 = g + lp
            checks = at_depth[depth] + (on_complete if depth == n - 1 else [])
            if checks:
                out = as_answer(child)
                if not passes(checks, out):
                    res.rejected_mass += math.exp(g2)
                    continue
            heapq.heappush(heap, (-(g2 + best_rest[depth + 1]), next(tie), g2, child))

    if pending and flush():
        return res
    if res.status != "exhausted":
        res.status = "impossible"
    return res
