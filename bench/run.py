"""Run rulebench v0 and compare three ways of getting a rule-following answer.

  A. model only    the decision model's own top answer (no rules)
  B. shishya       the same model call + Shishya's rule search
  C. llm + retry   an LLM writes the JSON answer; if it breaks a rule it is told which one and tries
                   again, up to 3 times (the usual validate-and-retry approach)

    python bench/make_tasks.py > bench/tasks.jsonl
    python bench/run.py --backend http://127.0.0.1:8009            # A + B with a decision server
    python bench/run.py --backend http://127.0.0.1:8009 --llm 30   # also C on the first 30 tasks

Writes bench/out/<label>.json (every answer) and prints a markdown table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shishya import RegoRules, Shishya
from shishya.backends import ChatLogprobsBackend, FakeBackend, SystemOneBackend, render
from shishya.rules import Rule
from shishya.schema import compile_schema

SCENARIOS = ("leave_request", "stock_alert", "shell_guard")


def load_scenario(name: str) -> dict:
    d = ROOT / "examples" / name
    sc = {"schema": json.loads((d / "schema.json").read_text()), "rules": [], "rego": None}
    if (d / "rules.py").exists():
        spec = importlib.util.spec_from_file_location(f"rules_{name}", d / "rules.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sc["rules"] = [v for v in vars(mod).values() if isinstance(v, Rule)]
    if (d / "policy.rego").exists():
        sc["rego"] = RegoRules((d / "policy.rego").read_text())
    return sc


def broken(sc: dict, ans: dict | None, ctx: dict) -> list[str]:
    """Names of the rules this complete answer breaks ([] when it follows all of them)."""
    if ans is None:
        return []
    out = [r.name for r in sc["rules"] if not r(ans, ctx)]
    if sc["rego"] is not None and not sc["rego"].check_batch([ans], ctx)[0][0]:
        out.append(sc["rego"].name)
    return out


def make_backend(spec: str):
    if spec == "fake":
        return FakeBackend()
    if spec == "openrouter":
        return ChatLogprobsBackend()
    if spec.startswith("openrouter:"):
        return ChatLogprobsBackend(model=spec.split(":", 1)[1])
    return SystemOneBackend(spec)


def run_llm_retry(guru, sc: dict, task: dict, max_tries: int = 3) -> dict:
    """C: the LLM proposes; on a broken rule or bad value it is told why and tries again."""
    comp = compile_schema(sc["schema"])
    ctx = {"input": task["input"], "facts": task["facts"]}
    spec = []
    for f in comp.fields:
        if f.when is not None:
            from shishya.when import holds
            if not holds(f.when, ctx):
                continue
        opts = [o for o in f.options if o not in f.option_when]
        from shishya.when import holds
        opts += [o for o in f.option_when if holds(f.option_when[o], ctx)]
        spec.append({"name": f.name, "kind": "boolean" if f.kind == "noul" else "choice",
                     "options": [o for o in f.options if o in opts], "instructions": f.instructions,
                     "option_text": dict(f.option_text)})
    names = [s["name"] for s in spec]
    feedback: list[str] = []
    first = None
    t0 = time.perf_counter()
    calls = 0
    ans = None
    for _ in range(max_tries):
        calls += 1
        try:
            got = guru.propose(render(task["input"], task["facts"]), spec, feedback)
        except Exception as e:  # noqa: BLE001
            return {"answer": None, "first": first, "calls": calls, "ms": (time.perf_counter() - t0) * 1000,
                    "error": str(e)[:200]}
        cand = {f.name: None for f in comp.fields}
        cand.update({k: got.get(k) for k in names})
        if first is None:
            first = cand
        bad = [f"{n!r} must be one of {s['options']}" for n, s in zip(names, spec) if cand[n] not in s["options"]]
        if not bad:
            msgs = {r.name: r.message for r in sc["rules"]}
            bad = [f"Not allowed: {msgs.get(b, b)}" for b in broken(sc, cand, ctx)]
        if not bad:
            ans = cand
            break
        feedback += [b for b in bad if b not in feedback]
    return {"answer": ans, "first": first, "calls": calls, "ms": (time.perf_counter() - t0) * 1000}


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r]
    n = len(vals)
    if not n:
        return {}
    right = sum(1 for v in vals if v["answer"] is not None and v["answer"].get(v["field"]) == v["gold"])
    answered = sum(1 for v in vals if v["answer"] is not None)
    viol = sum(1 for v in vals if v["broken"])
    ms = [v["ms"] for v in vals if v.get("ms") is not None]
    return {"n": n, "accuracy": right / n, "answered": answered / n, "violations": viol / n,
            "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95),
            "llm_calls": sum(v.get("calls", 0) for v in vals) / n}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "bench" / "tasks.jsonl"))
    ap.add_argument("--backend", default="http://127.0.0.1:8009")
    ap.add_argument("--label", default=None)
    ap.add_argument("--llm", type=int, default=0, help="also run C (LLM + retry) on the first N tasks")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args(argv)

    tasks = [json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    scen = {name: load_scenario(name) for name in SCENARIOS}
    backend = make_backend(args.backend)
    engines = {name: Shishya(backend, rules=sc["rules"], rego=sc["rego"]) for name, sc in scen.items()}
    guru = None
    if args.llm:
        from shishya.guru import OpenRouterGuru
        guru = OpenRouterGuru(model=args.llm_model) if args.llm_model else OpenRouterGuru()

    rows = []
    for i, t in enumerate(tasks):
        sc, eng = scen[t["scenario"]], engines[t["scenario"]]
        ctx = {"input": t["input"], "facts": t["facts"]}
        t0 = time.perf_counter()
        d = eng.decide(t["input"], sc["schema"], facts=t["facts"])
        ms = (time.perf_counter() - t0) * 1000
        model_ms = d.receipt["timings_ms"]["model_ms"]
        row = {"id": t["id"], "kind": t["kind"], "gold": t["gold"],
               "A": {"answer": d.top, "field": t["field"], "gold": t["gold"], "broken": broken(sc, d.top, ctx),
                     "ms": model_ms, "calls": 0},
               "B": {"answer": d.answer, "field": t["field"], "gold": t["gold"], "status": d.status,
                     "broken": broken(sc, d.answer, ctx), "ms": ms, "tried": d.tried, "calls": 0}}
        if guru is not None and i < args.llm:
            c = run_llm_retry(guru, sc, t)
            c.update({"field": t["field"], "gold": t["gold"], "broken": broken(sc, c["answer"], ctx),
                      "first_broken": broken(sc, c["first"], ctx) if c.get("first") else []})
            row["C"] = c
        rows.append(row)
        print(f"{i + 1:>3}/{len(tasks)} {t['id']:<10} gold={t['gold']:<9} A={(d.top or {}).get(t['field'])!s:<9} "
              f"B={(d.answer or {}).get(t['field'])!s:<9} {d.status:<6} {ms:6.0f} ms"
              + (f"  C={(row['C']['answer'] or {}).get(t['field'])!s}" if "C" in row else ""), file=sys.stderr)

    label = args.label or (getattr(backend, "model", None) or args.backend).replace("/", "_").replace(":", "_")
    out_dir = ROOT / "bench" / "out"
    out_dir.mkdir(exist_ok=True)
    summary = {k: summarize(rows, k) for k in ("A", "B", "C")}
    result = {"backend": args.backend, "model": getattr(backend, "model", None), "summary": summary, "rows": rows}
    (out_dir / f"{label}.json").write_text(json.dumps(result, indent=1, default=str))
    names = {"A": "model only (no rules)", "B": "**Shishya** (model + rule search)", "C": "LLM writes JSON + retry ≤3"}
    print(f"\nBackend: `{args.backend}` (model `{getattr(backend, 'model', None)}`), {len(tasks)} tasks\n")
    print("| System | Tasks | Correct | Breaks a rule | Answered | p50 ms | p95 ms | LLM calls/task |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, s in summary.items():
        if s:
            print(f"| {names[k]} | {s['n']} | {s['accuracy']:.0%} | {s['violations']:.0%} | {s['answered']:.0%} | "
                  f"{s['p50_ms']:.0f} | {s['p95_ms']:.0f} | {s['llm_calls']:.1f} |")


if __name__ == "__main__":
    main()
