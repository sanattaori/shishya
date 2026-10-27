"""Calibration: fit one temperature per field so the model's probabilities can be trusted.

Decision models are often over- or under-confident. Given your own labelled examples, this fits a
temperature T per field (probabilities become p^(1/T), renormalized: T > 1 softens an over-confident
model, T < 1 sharpens an under-confident one) and reports, honestly, how many decisions could be
automated at a given confidence and what error that gives. The report uses 2-fold cross-fitting: T is
fitted on one half of the examples and measured on the other half, so the error is never measured on
the data T was fitted on.

Labelled examples are JSONL, one object per line:

    {"input": "Need 5 days off", "facts": {"leave_balance": 3}, "answer": {"verdict": "deny", "urgent": false}}

Only the fields you labelled need to be in ``answer``.

    collected = collect(backend, schema, load_labels("labels.jsonl"))
    temps = fit(collected)
    print(report(collected, temps))
    engine = Shishya(CalibratedBackend(backend, temps))
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import render
from .schema import compile_schema
from .when import holds

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
GRID = [math.exp(math.log(0.25) + (math.log(8) - math.log(0.25)) * i / 399) for i in range(400)]
_FLOOR = 1e-12
FORMAT = 1


def load_labels(path: str) -> list[dict]:
    """Read a JSONL file of labelled examples (blank lines are ignored)."""
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {n}: not valid JSON ({e})") from e
        if not isinstance(ex, dict) or "input" not in ex or not isinstance(ex.get("answer"), dict):
            raise ValueError(f"{path} line {n}: each line needs an 'input' and an 'answer' object")
        out.append(ex)
    if not out:
        raise ValueError(f"{path} has no labelled examples")
    return out


def _normalize(probs: dict) -> dict:
    total = sum(max(float(v), 0.0) for v in probs.values())
    if total <= 0:
        return {k: 1.0 / len(probs) for k in probs}
    return {k: max(float(v), 0.0) / total for k, v in probs.items()}


def collect(backend: Any, schema: dict, examples: list[dict], show_facts: bool = True) -> list[dict]:
    """Score every example once and keep, per labelled field, the option probabilities and the gold value.

    Returns one entry per example that has at least one usable label:
    ``{"index", "fields": {name: {"probs": {option: p}, "gold": value}}, "skipped": n}``.
    A field is used only when the engine would ask the model about it for this example (its ``when`` holds
    and at least two options are allowed). Labels that are not an allowed option are skipped and counted.
    """
    comp = compile_schema(schema)
    out = []
    for i, ex in enumerate(examples):
        labels = ex.get("answer") or {}
        facts = ex.get("facts")
        ctx = {"input": ex.get("input"), "facts": facts}
        unknown = sorted(set(labels) - set(comp.names))
        if unknown:
            raise ValueError(f"example {i + 1}: the schema has no field {unknown}")
        active, options, skipped = [], {}, 0
        for f in comp.fields:
            if f.name not in labels:
                continue
            if f.when is not None and not holds(f.when, ctx):
                skipped += 1
                continue
            opts = tuple(o for o in f.options if o not in f.option_when or holds(f.option_when[o], ctx))
            if len(opts) < 2:
                skipped += 1
                continue
            gold = labels[f.name]
            right_type = isinstance(gold, bool) if f.kind == "noul" else isinstance(gold, str)
            if not right_type or gold not in opts:
                skipped += 1
                continue
            active.append(f)
            options[f.name] = opts
        if not active:
            continue
        probs = backend.score(render(ex.get("input"), facts, show_facts), active, options)
        out.append({"index": i, "skipped": skipped, "fields": {
            f.name: {"probs": _normalize(probs[f.name]), "gold": labels[f.name]} for f in active}})
    return out


def apply(probs: dict, T: float) -> dict:
    """p^(1/T), renormalized. Options at zero stay at zero."""
    if T <= 0:
        raise ValueError("temperature must be positive")
    live = {k: math.log(v) / T for k, v in probs.items() if v > 0}
    if not live:
        return dict(probs)
    top = max(live.values())
    w = {k: math.exp(x - top) for k, x in live.items()}
    total = sum(w.values())
    return {k: (w[k] / total if k in w else 0.0) for k in probs}


def _fit_one(rows: list[tuple[dict, Any]]) -> float:
    logs = [({k: math.log(max(v, _FLOOR)) for k, v in p.items()}, g) for p, g in rows]
    best_t, best = 1.0, math.inf
    for T in GRID:
        nll = 0.0
        for lp, gold in logs:
            scaled = [x / T for x in lp.values()]
            top = max(scaled)
            lse = top + math.log(sum(math.exp(x - top) for x in scaled))
            nll += lse - lp[gold] / T
        if nll < best:
            best_t, best = T, nll
    return best_t


def _rows(collected: list[dict]) -> dict[str, list[tuple[dict, Any]]]:
    rows: dict[str, list[tuple[dict, Any]]] = {}
    for ex in collected:
        for name, d in ex["fields"].items():
            rows.setdefault(name, []).append((d["probs"], d["gold"]))
    return rows


def fit(collected: list[dict]) -> dict[str, float]:
    """Per field, the temperature that minimizes the negative log-likelihood of the gold answers."""
    return {name: _fit_one(rows) for name, rows in _rows(collected).items()}


def _top(probs: dict) -> tuple[Any, float]:
    k = max(probs, key=probs.get)
    return k, probs[k]


def _ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Expected calibration error: gap between stated confidence and actual accuracy, averaged over bins."""
    if not pairs:
        return 0.0
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in pairs:
        groups[min(int(conf * bins), bins - 1)].append((conf, ok))
    n = len(pairs)
    return sum(abs(sum(ok for _, ok in g) / len(g) - sum(c for c, _ in g) / len(g)) * len(g) / n
               for g in groups if g)


def _table(items: list[tuple[float, bool]], thresholds: tuple[float, ...]) -> list[dict]:
    """items: (confidence, correct) per example. Coverage and error among the covered, per threshold."""
    rows = []
    for t in thresholds:
        kept = [ok for conf, ok in items if conf >= t]
        rows.append({"threshold": t, "covered": len(kept),
                     "coverage": len(kept) / len(items) if items else 0.0,
                     "error": (kept.count(False) / len(kept)) if kept else None})
    return rows


def report(collected: list[dict], temps: dict[str, float], thresholds: tuple[float, ...] = THRESHOLDS) -> dict:
    """Honest numbers: calibrated results come from 2-fold cross-fitting (split by example position parity)."""
    halves = [[ex for k, ex in enumerate(collected) if k % 2 == h] for h in (0, 1)]
    fold_temps = [fit(halves[1 - h]) for h in (0, 1)]     # temperatures fitted on the other half
    per_field: dict[str, list[tuple[float, bool, float, bool]]] = {}   # before conf/ok, after conf/ok
    joint: list[tuple[float, bool]] = []
    for h in (0, 1):
        for ex in halves[h]:
            worst, all_ok = 1.0, True
            for name, d in ex["fields"].items():
                raw_pick, raw_conf = _top(d["probs"])
                cal = apply(d["probs"], fold_temps[h].get(name, 1.0))
                pick, conf = _top(cal)
                per_field.setdefault(name, []).append((raw_conf, raw_pick == d["gold"], conf, pick == d["gold"]))
                worst, all_ok = min(worst, conf), all_ok and pick == d["gold"]
            joint.append((worst, all_ok))
    fields = {}
    for name, items in per_field.items():
        fields[name] = {
            "n": len(items), "temperature": temps.get(name),
            "accuracy": sum(ok for _, ok, _, _ in items) / len(items),
            "ece_before": _ece([(c, ok) for c, ok, _, _ in items]),
            "ece_after": _ece([(c, ok) for _, _, c, ok in items]),
            "thresholds": _table([(c, ok) for _, _, c, ok in items], thresholds)}
    return {"n_examples": len(collected), "skipped_labels": sum(ex["skipped"] for ex in collected),
            "thresholds": list(thresholds), "fields": fields,
            "all_fields": {"n": len(joint), "thresholds": _table(joint, thresholds)}}


class CalibratedBackend:
    """Wrap any backend and apply each field's temperature to its probabilities."""

    name = "calibrated"

    def __init__(self, backend: Any, temps: dict[str, float]):
        self.backend = backend
        self.temps = dict(temps)

    @property
    def model(self) -> str:
        return f"{getattr(self.backend, 'model', self.backend.name)} (calibrated)"

    def score(self, text: str, fields: list, options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        out = self.backend.score(text, fields, options)
        return {name: (apply(p, self.temps[name]) if name in self.temps else p) for name, p in out.items()}


def save(path: str, temps: dict[str, float], meta: dict | None = None) -> None:
    body = {"shishya_calibration": FORMAT, "temperatures": temps,
            "meta": {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(meta or {})}}
    Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")


def load(path: str) -> dict[str, float]:
    try:
        body = json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON ({e})") from e
    temps = body.get("temperatures") if isinstance(body, dict) else None
    if not isinstance(body, dict) or body.get("shishya_calibration") != FORMAT or not isinstance(temps, dict):
        raise ValueError(f"{path} is not a Shishya calibration file (made by 'shishya calibrate')")
    try:
        out = {str(k): float(v) for k, v in temps.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path}: temperatures must be numbers") from e
    if any(v <= 0 for v in out.values()):
        raise ValueError(f"{path}: temperatures must be positive")
    return out
