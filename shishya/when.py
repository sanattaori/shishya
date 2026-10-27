"""Conditions over the request context, evaluated by plain code (never by the model).

A condition is JSON:

    {"path": "facts.leave_balance", "gte": 3}
    {"path": "facts.holdings", "contains": "TCS"}
    {"exists": "facts.manager_ok"}
    {"all": [cond, ...]}   {"any": [cond, ...]}   {"not": cond}

Leaf operators: eq, neq, gt, gte, lt, lte, in (value is a list), contains (path is a list or string).
Paths are dotted and start at ``input`` (what the model reads), ``facts`` (what your app knows) or, in
rules only, ``answer`` (the answer being checked).
List items are addressed by index (``facts.trades.0.qty``).

Missing facts are *unknown*, not false: a comparison with a missing path is unknown, ``not unknown`` is
unknown, and the final unknown counts as "no". So a field or option guarded by a fact the app did not
send is never made eligible by accident.
"""
from __future__ import annotations

from typing import Any

_MISSING = object()
LEAF_OPS = ("eq", "neq", "gt", "gte", "lt", "lte", "in", "contains")
ROOTS = ("input", "facts", "answer")


class ConditionError(ValueError):
    pass


def lookup(path: str, ctx: dict) -> Any:
    """Value at a dotted path in the context, or _MISSING."""
    if not isinstance(path, str) or not path:
        raise ConditionError(f"path must be a non-empty string (got {path!r})")
    parts = path.split(".")
    if parts[0] not in ROOTS:
        raise ConditionError(f"path {path!r} must start with one of {', '.join(ROOTS)}")
    cur: Any = ctx
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return _MISSING
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            i = int(part)
            if i >= len(cur):
                return _MISSING
            cur = cur[i]
        else:
            return _MISSING
    return cur


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _compare(op: str, left: Any, right: Any) -> bool | None:
    if op == "eq":
        return left == right
    if op == "neq":
        return left != right
    if op in ("gt", "gte", "lt", "lte"):
        # numbers with numbers, strings with strings; anything else is unknown (no coercion)
        if not ((_is_number(left) and _is_number(right)) or (isinstance(left, str) and isinstance(right, str))):
            return None
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[op]
    if op == "in":
        if not isinstance(right, list):
            raise ConditionError("'in' needs a list value")
        return left in right
    if op == "contains":
        if isinstance(left, list):
            return right in left
        if isinstance(left, str) and isinstance(right, str):
            return right in left
        return None
    raise ConditionError(f"unknown operator {op!r}")


def evaluate(cond: Any, ctx: dict) -> bool | None:
    """True, False or None (unknown)."""
    if not isinstance(cond, dict) or not cond:
        raise ConditionError(f"condition must be a non-empty object (got {cond!r})")
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        items = cond[key]
        if len(cond) != 1 or not isinstance(items, list) or not items:
            raise ConditionError(f"'{key}' must be the only key and hold a non-empty list")
        vals = [evaluate(c, ctx) for c in items]
        if key == "all":
            if any(v is False for v in vals):
                return False
            return None if any(v is None for v in vals) else True
        if any(v is True for v in vals):
            return True
        return None if any(v is None for v in vals) else False
    if "not" in cond:
        if len(cond) != 1:
            raise ConditionError("'not' must be the only key")
        v = evaluate(cond["not"], ctx)
        return None if v is None else not v
    if "exists" in cond:
        if len(cond) != 1:
            raise ConditionError("'exists' must be the only key")
        return lookup(cond["exists"], ctx) is not _MISSING
    if "path" not in cond:
        raise ConditionError(f"condition needs 'path' or one of all/any/not/exists (got {sorted(cond)})")
    ops = [k for k in cond if k != "path"]
    if len(ops) != 1 or ops[0] not in LEAF_OPS:
        raise ConditionError(f"a leaf condition needs exactly one operator from {', '.join(LEAF_OPS)} (got {ops})")
    left = lookup(cond["path"], ctx)
    if left is _MISSING:
        return None
    return _compare(ops[0], left, cond[ops[0]])


def holds(cond: Any, ctx: dict) -> bool:
    """Condition is definitely true (unknown counts as false)."""
    return evaluate(cond, ctx) is True


def check(cond: Any) -> None:
    """Validate a condition's shape without a context (raises ConditionError)."""
    if not isinstance(cond, dict) or not cond:
        raise ConditionError(f"condition must be a non-empty object (got {cond!r})")
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        if len(cond) != 1 or not isinstance(cond[key], list) or not cond[key]:
            raise ConditionError(f"'{key}' must be the only key and hold a non-empty list")
        for c in cond[key]:
            check(c)
        return
    if "not" in cond:
        if len(cond) != 1:
            raise ConditionError("'not' must be the only key")
        check(cond["not"])
        return
    if "exists" in cond:
        if len(cond) != 1:
            raise ConditionError("'exists' must be the only key")
        lookup(cond["exists"], {})
        return
    if "path" not in cond:
        raise ConditionError(f"condition needs 'path' or one of all/any/not/exists (got {sorted(cond)})")
    ops = [k for k in cond if k != "path"]
    if len(ops) != 1 or ops[0] not in LEAF_OPS:
        raise ConditionError(f"a leaf condition needs exactly one operator from {', '.join(LEAF_OPS)} (got {ops})")
    if ops[0] == "in" and not isinstance(cond["in"], list):
        raise ConditionError("'in' needs a list value")
    lookup(cond["path"], {})


def paths(cond: Any) -> list[str]:
    """Every path a (valid) condition reads, in order."""
    if "all" in cond or "any" in cond:
        return [p for c in cond.get("all", cond.get("any")) for p in paths(c)]
    if "not" in cond:
        return paths(cond["not"])
    if "exists" in cond:
        return [cond["exists"]]
    return [cond["path"]]
