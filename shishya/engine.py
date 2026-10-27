"""Shishya: the most likely answer that follows your rules, or an honest "not sure".

    from shishya import Shishya, rule
    from shishya.backends import SystemOneBackend

    @rule(uses=["verdict"])
    def within_balance(ans, ctx):
        "Leave can be approved only up to the remaining balance."
        return ans["verdict"] != "approve" or ctx["facts"]["days_asked"] <= ctx["facts"]["leave_balance"]

    s = Shishya(SystemOneBackend("http://127.0.0.1:8009"), rules=[within_balance])
    d = s.decide("Need 5 days off next week for my sister's wedding", schema,
                 facts={"days_asked": 5, "leave_balance": 3})
    d.status, d.answer, d.probability

Steps: compile the schema (cached) -> conditions decide which fields/options apply (plain code) -> the
model scores every allowed option in one call -> best-first search over the rules -> if nothing
passes or confidence is too low, ask the Guru (an LLM) with the broken rules as feedback, or report
"unsure" / "none".
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from . import __version__
from .backends.base import Backend, render
from .rules import RegoRules, Rule, as_rules
from .schema import Compiled, compile_schema, digest
from .search import normalize, search
from .when import holds, unknown_facts


def fact_notes(rules: list[Rule], ctx: dict) -> tuple[list[str], list[str]]:
    """Facts that leave JSON rules unknown: (missing fact paths, notes saying what each rule did about it)."""
    missing, blocking, skipped, notes = set(), set(), set(), []
    for r in rules:
        if not r.spec:
            continue
        cond = r.spec.get("forbid", r.spec.get("require"))
        lenient = r.spec.get("on_missing") == "allow"
        for path, why in unknown_facts(cond, ctx).items():
            if why == "missing":
                missing.add(path)
                (skipped if lenient else blocking).add(path)
            else:
                notes.append(f"rule {r.name!r} cannot use {path}: it {why}, so the rule "
                             + ("is skipped" if lenient else "blocks the answers it might forbid"))
    if blocking:
        notes.insert(0, "facts the rules need were not sent, so those rules block: " + ", ".join(sorted(blocking)))
    if skipped:
        notes.insert(0, "facts were not sent, so rules marked on_missing=allow skip them: "
                     + ", ".join(sorted(skipped)))
    return sorted(missing), notes


def _fail_closed(rule: Rule, errors: list[str]) -> Rule:
    """The same rule, except that an exception blocks the answer (and is reported) instead of crashing."""
    def fn(ans: dict, ctx: dict) -> bool:
        try:
            return bool(rule.fn(ans, ctx))
        except Exception as e:  # noqa: BLE001 - any failure inside a rule means "not allowed"
            errors.append(f"{rule.name}: {type(e).__name__}: {e}")
            return False
    return Rule(fn, rule.uses, rule.name, rule.message, rule.spec)

STATUSES = ("ok", "guru", "unsure", "none")


class Guru(Protocol):
    name: str
    model: str
    max_tries: int

    def propose(self, text: str, fields: list[dict], feedback: list[str]) -> dict:
        """One JSON answer for ``fields`` (name, kind, options, instructions); may raise."""
        ...


@dataclass
class Decision:
    status: str                    # ok | guru | unsure | none
    answer: dict | None            # the answer (set when status is ok or guru)
    probability: float             # joint model probability of the answer (0 for guru answers)
    confidence: float = 0.0        # the weakest field: lowest model probability among the decided fields
    fields: dict = field(default_factory=dict)       # per field: {"value", "probability", "options"}
    top: dict | None = None        # the model's own most likely answer, before the rules
    top_allowed: bool | None = None
    suggestion: dict | None = None  # unsure: the best allowed answer, below the confidence bar
    tried: int = 0                 # complete answers the search checked
    rejected_mass: float = 0.0     # model probability the rules ruled out
    broken_rules: dict = field(default_factory=dict)  # rule name -> answers it rejected
    reasons: list = field(default_factory=list)       # policy reasons and notes
    trainable: bool = True         # False when the answer came from a Guru whose outputs must not train models
    missing_facts: list = field(default_factory=list)  # facts the rules needed but were not sent
    rule_errors: list = field(default_factory=list)    # rules that raised: they block, never allow
    receipt: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Shishya:
    def __init__(self, backend: Backend, *, rules: Any = (), rego: RegoRules | str | None = None,
                 guru: Guru | None = None, max_candidates: int = 64, min_probability: float = 0.0,
                 show_facts: bool = True):
        self.backend = backend
        self.rules: list[Rule] = as_rules(rules)
        self.rego = RegoRules(rego) if isinstance(rego, str) else rego
        self.guru = guru
        self.max_candidates = max_candidates
        self.min_probability = min_probability
        self.show_facts = show_facts
        self._cache: dict[str, Compiled] = {}

    def compile(self, schema: dict) -> Compiled:
        key = digest(schema)
        if key not in self._cache:
            self._cache[key] = compile_schema(schema)
        return self._cache[key]

    def decide(self, input: Any, schema: dict, *, facts: Any = None, rules: Any = None,
               rego: RegoRules | str | None = None) -> Decision:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        comp = self.compile(schema)
        errors: list[str] = []
        user_rules = list(comp.rules) + (as_rules(rules) if rules is not None else self.rules)
        rule_list = [_fail_closed(r, errors) for r in user_rules]
        rego_obj = (RegoRules(rego) if isinstance(rego, str) else rego) if rego is not None else self.rego
        checkers = [rego_obj] if rego_obj is not None else []
        ctx = {"input": input, "facts": facts}

        # 1. conditions (plain code): which fields are asked, which options are allowed
        fixed: dict[str, Any] = {}
        active, options = [], {}
        for f in comp.fields:
            if f.when is not None and not holds(f.when, ctx):
                fixed[f.name] = None
                continue
            opts = tuple(o for o in f.options if o not in f.option_when or holds(f.option_when[o], ctx))
            if not opts:
                fixed[f.name] = None
            elif len(opts) == 1:
                fixed[f.name] = opts[0]
            else:
                active.append(f)
                options[f.name] = opts

        # 2. the model scores every allowed option, in one call
        text = render(input, facts, self.show_facts)
        t = time.perf_counter()
        probs = self.backend.score(text, active, options) if active else {}
        timings["model_ms"] = (time.perf_counter() - t) * 1000
        probs = {name: normalize(p) for name, p in probs.items()}

        def in_order(ans: dict | None) -> dict | None:
            return None if ans is None else {n: ans.get(n) for n in comp.names}

        top = dict(fixed)
        top.update({name: max(p, key=p.get) for name, p in probs.items()})
        top = in_order(top)

        # 3. best-first search over the rules
        t = time.perf_counter()
        res = search([f.name for f in active], probs, rules=rule_list, checkers=checkers, ctx=ctx, fixed=fixed,
                     max_candidates=self.max_candidates)
        timings["search_ms"] = (time.perf_counter() - t) * 1000

        def field_report(ans: dict | None) -> dict:
            rep = {}
            for f in comp.fields:
                p = probs.get(f.name)
                v = ans.get(f.name) if ans else None
                if p is not None:
                    prob = p.get(v)
                elif f.name in fixed and v is not None:
                    prob = 1.0
                else:
                    prob = None
                rep[f.name] = {"value": v, "probability": prob,
                               "options": ({(str(k).lower() if isinstance(k, bool) else k): round(x, 6)
                                            for k, x in p.items()} if p else None)}
            return rep

        d = Decision(status="none", answer=None, probability=0.0, top=top, tried=res.tried,
                     rejected_mass=round(res.rejected_mass, 6), broken_rules=dict(res.failures),
                     reasons=list(res.reasons))
        d.missing_facts, notes = fact_notes(list(comp.rules), ctx)
        d.reasons.extend(notes)
        best = in_order(res.answer)
        d.top_allowed = best is not None and best == top
        if best is not None:
            # every field the model decided must clear the bar on its own (a product over fields would
            # fall below any bar as soon as there are a few fields)
            d.confidence = min((probs[n][best[n]] for n in probs), default=1.0)
        if best is not None and d.confidence >= self.min_probability:
            d.status, d.answer, d.probability = "ok", best, res.probability
        elif best is not None:
            d.status, d.suggestion, d.probability = "unsure", best, res.probability
        d.fields = field_report(d.answer or d.suggestion)

        # 4. fallback: the Guru answers with the broken rules as feedback and must pass the same checks
        if d.status != "ok" and self.guru is not None:
            t = time.perf_counter()
            got = self._ask_guru(text, active, options, fixed, rule_list, checkers, ctx, d)
            timings["guru_ms"] = (time.perf_counter() - t) * 1000
            if got is not None:
                got = in_order(got)
                d.status, d.answer, d.probability, d.confidence, d.suggestion = "guru", got, 0.0, 0.0, None
                d.trainable = bool(getattr(self.guru, "trainable", False))
                d.fields = field_report(got)
        if d.status == "none" and res.status == "exhausted":
            d.reasons.append(f"search budget reached after {res.tried} answers")

        d.rule_errors = sorted(set(errors))
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        d.receipt = {
            "shishya": __version__,
            "schema_digest": comp.digest,
            "rules": [{"name": r.name, "fingerprint": r.fingerprint()} for r in user_rules]
                     + [{"name": c.name, "fingerprint": c.fingerprint()} for c in checkers],
            "backend": {"name": self.backend.name, "model": getattr(self.backend, "model", None)},
            "guru": ({"name": self.guru.name, "model": self.guru.model} if self.guru else None),
            "search": {"status": res.status, "expanded": res.expanded, "tried": res.tried},
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }
        return d

    # -- guru ---------------------------------------------------------------------------------------
    def _ask_guru(self, text, active, options, fixed, rule_list, checkers, ctx, d: Decision) -> dict | None:
        spec = [{"name": f.name, "kind": "boolean" if f.kind == "noul" else "choice",
                 "options": list(options[f.name]), "instructions": f.instructions,
                 "option_text": {o: f.option_text[o] for o in options[f.name] if o in f.option_text}}
                for f in active]
        names = {f.name for f in active}
        messages = {r.name: r.message for r in rule_list}
        feedback = [f"Not allowed: {messages.get(n, n)}" for n in d.broken_rules] + list(d.reasons)
        for _ in range(max(1, self.guru.max_tries)):
            try:
                got = self.guru.propose(text, spec, feedback)
            except Exception as e:          # noqa: BLE001 - a Guru failure must not break the decision
                d.reasons.append(f"guru error: {e}")
                return None
            problems = []
            if not isinstance(got, dict):
                problems.append("the answer must be a JSON object")
                got = {}
            for name in sorted(names):
                if name not in got:
                    problems.append(f"missing field {name!r}")
                elif got[name] not in options[name]:
                    problems.append(f"{name!r} must be one of {list(options[name])}")
            extra = set(got) - names
            if extra:
                problems.append(f"unknown fields {sorted(extra)}")
            if not problems:
                ans = dict(fixed)
                ans.update({k: got[k] for k in names})
                problems += [f"Not allowed: {r.message}" for r in rule_list if not r(ans, ctx)]
                for chk in checkers:
                    ok, why = chk.check_batch([ans], ctx)[0]
                    if not ok:
                        problems.append(f"Not allowed by {chk.name}" + (f": {'; '.join(why)}" if why else ""))
                if not problems:
                    return ans
            feedback = feedback + [p for p in problems if p not in feedback]
        d.reasons.append("the Guru could not find an allowed answer")
        return None
