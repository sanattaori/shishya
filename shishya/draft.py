"""Draft rules from plain English, safely.

A Guru (an LLM) DRAFTS declarative JSON rules (the format of ``rules.json_rule``). Nothing is applied:
every drafted rule is checked by ``json_rule`` (no code can run), shown to you with examples of the
answers it blocks and allows, and only you add it to the schema under ``x-shishya.rules``.

    rules = draft_rules(guru, schema, "Never buy while the market guard is on.", facts_example={...})
    rows = examples(schema, rules, facts={"market_guard_on": True})
"""
from __future__ import annotations

import itertools
import json
import re
from typing import Any

from .rules import RuleError, json_rule
from .schema import compile_schema
from .when import holds

MAX_COMBINATIONS = 2000

REPLY_SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "message": {"type": "string"},
                       "forbid": {"type": "object"}, "require": {"type": "object"}},
        "required": ["name", "message"]}}},
    "required": ["rules"],
}

_RULE_FORMAT = """You turn a person's plain-English rules into declarative JSON rules for a decision engine.
You only DRAFT: a human reviews every rule before it is used.

A rule is a JSON object with exactly these keys:
  "name"    a short snake_case identifier, unique among the rules
  "message" the person's sentence for this rule, in plain words
  "forbid" OR "require" (exactly one): a condition

forbid: the answer is rejected when the condition is definitely true.
require: the answer is rejected unless the condition is definitely true.
Unknown facts (a fact that was not provided) never make a forbid rule fire, and always make a require rule fail.
Prefer forbid unless the sentence says something must be true.

A condition is a JSON object, one of:
  {"path": P, "OP": value}   with OP one of eq, neq, gt, gte, lt, lte, in (value is a list), contains
                             (P is a list or string and the value is looked for inside it)
  {"exists": P}
  {"all": [condition, ...]}  every condition holds
  {"any": [condition, ...]}  at least one holds
  {"not": condition}
Paths: answer.<field> (the answer being checked), facts.<key> (what the app knows), input (the text).
Dotted paths go deeper (facts.trades.0.qty). Comparisons gt/gte/lt/lte work on numbers with numbers or strings
with strings only. A condition can only compare a path with a fixed value, never with another path.
Every rule must read at least one answer.<field>. Booleans are JSON true/false.
Use only the fields, options and fact keys listed below; do not invent any. Reply with JSON only."""


class DraftError(ValueError):
    """The drafted rules could not be made valid."""


def system_prompt(schema: dict, facts_example: dict | None = None, existing: list[str] | None = None) -> str:
    comp = compile_schema(schema)
    lines = [_RULE_FORMAT, "", "Answer fields:"]
    for f in comp.fields:
        shown = ", ".join("true" if o is True else "false" if o is False else json.dumps(o) for o in f.options)
        lines.append(f"- answer.{f.name} ({f.instructions}): one of {shown}")
    if facts_example:
        lines += ["", "Fact keys (with an example value):"]
        lines += [f"- facts.{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in facts_example.items()]
    else:
        lines += ["", "No facts were described; use facts.<key> only when the sentence clearly names a fact."]
    if existing:
        lines += ["", "Rule names already taken (do not reuse): " + ", ".join(existing)]
    return "\n".join(lines)


def _bad_values(cond: Any, options: dict[str, tuple]) -> list[str]:
    """Values compared with an answer field that the field can never take (the rule would be inert)."""
    problems = []
    if "all" in cond or "any" in cond:
        for c in cond.get("all", cond.get("any")):
            problems += _bad_values(c, options)
    elif "not" in cond:
        problems += _bad_values(cond["not"], options)
    elif "path" in cond:
        parts = cond["path"].split(".")
        if parts[0] == "answer" and len(parts) == 2 and parts[1] in options:
            for op in ("eq", "neq", "in"):
                if op in cond:
                    vals = cond[op] if op == "in" else [cond[op]]
                    problems += [f"{cond['path']} can never be {v!r} (options: {list(options[parts[1]])})"
                                 for v in vals if v not in options[parts[1]]]
    return problems


def _unknown_facts(cond: Any, keys: set[str]) -> list[str]:
    """facts.<key> paths whose top-level key is not among the described facts (the rule could never fire)."""
    from .when import paths
    return sorted({p for p in paths(cond) if p.split(".")[0] == "facts" and len(p.split(".")) > 1
                   and p.split(".")[1] not in keys})


def _validate(reply: Any, schema: dict, facts_example: dict | None = None) -> tuple[list[dict], list[str]]:
    comp = compile_schema(schema)
    names = set(comp.names)
    options = {f.name: f.options for f in comp.fields}
    if not isinstance(reply, dict) or not isinstance(reply.get("rules"), list) or not reply["rules"]:
        return [], ['the reply must be a JSON object {"rules": [ ... ]} with at least one rule']
    problems, seen, good = [], {r.name for r in comp.rules}, []
    for i, spec in enumerate(reply["rules"], 1):
        label = spec.get("name", f"#{i}") if isinstance(spec, dict) else f"#{i}"
        try:
            json_rule(spec, names)
        except RuleError as e:
            problems.append(str(e))
            continue
        extra = []
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec["name"]):
            extra.append(f"rule {label!r}: the name must be snake_case")
        if spec["name"] in seen:
            extra.append(f"rule {label!r}: the name is already used")
        seen.add(spec["name"])
        cond = spec.get("forbid", spec.get("require"))
        extra += [f"rule {label!r}: {p}" for p in _bad_values(cond, options)]
        if facts_example:
            extra += [f"rule {label!r}: {p} is not one of the described facts ({sorted(facts_example)})"
                      for p in _unknown_facts(cond, set(facts_example))]
        if extra:
            problems += extra
        else:
            good.append(spec)
    return good, problems


def draft_rules(guru: Any, schema: dict, english: str, facts_example: dict | None = None) -> list[dict]:
    """Ask the Guru for rules, check every one with ``json_rule``, retry once with the errors as feedback."""
    if not isinstance(english, str) or not english.strip():
        raise DraftError("write the rules in plain English first")
    comp = compile_schema(schema)
    system = system_prompt(schema, facts_example, [r.name for r in comp.rules])
    prompt = f"Draft rules for this text:\n\n{english.strip()}"
    reply = guru.complete_json(system, prompt, REPLY_SCHEMA)
    good, problems = _validate(reply, schema, facts_example)
    if problems:
        feedback = (f"{prompt}\n\nYour previous draft was:\n{json.dumps(reply, ensure_ascii=False)}\n\n"
                    "It had these problems. Fix all of them and reply with the complete corrected list:\n"
                    + "\n".join(f"- {p}" for p in problems))
        reply = guru.complete_json(system, feedback, REPLY_SCHEMA)
        good, problems = _validate(reply, schema, facts_example)
    if problems:
        raise DraftError("the drafted rules are not valid, even after one retry:\n"
                         + "\n".join(f"- {p}" for p in problems))
    return good


def examples(schema: dict, rules: list[dict], facts: dict, limit: int = 12) -> list[dict]:
    """Illustrative complete answers for ``facts``: rows ``{"answer", "blocked_by": [rule names]}``.

    Every complete answer is enumerated (fields and options the schema's conditions allow for these facts;
    at most 2,000 combinations, the rest are skipped). For each rule, one answer it blocks is shown and the
    most similar answer that nothing blocks.
    """
    comp = compile_schema(schema)
    compiled = [json_rule(r, set(comp.names)) for r in rules]
    ctx = {"input": None, "facts": facts}
    choices = []
    for f in comp.fields:
        if f.when is not None and not holds(f.when, ctx):
            choices.append([None])
            continue
        opts = [o for o in f.options if o not in f.option_when or holds(f.option_when[o], ctx)]
        choices.append(opts or [None])
    rows = []
    for combo in itertools.islice(itertools.product(*choices), MAX_COMBINATIONS):
        ans = dict(zip(comp.names, combo))
        rows.append({"answer": ans, "blocked_by": [r.name for r in compiled if not r(ans, ctx)]})
    allowed = [r for r in rows if not r["blocked_by"]]

    def distance(a: dict, b: dict) -> int:
        return sum(a["answer"][k] != b["answer"][k] for k in a["answer"])

    picked: list[dict] = []

    def add(row: dict) -> None:
        if not any(row is p for p in picked) and len(picked) < limit:
            picked.append(row)

    for r in compiled:
        blocked = next((x for x in rows if r.name in x["blocked_by"]), None)
        if blocked is None:
            continue
        add(blocked)
        if allowed:
            add(min(allowed, key=lambda x: distance(x, blocked)))
    if allowed and not any(not p["blocked_by"] for p in picked):
        add(allowed[0])
    return picked
