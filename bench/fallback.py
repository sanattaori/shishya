"""Compare Shishya with the simplest alternative: check the model's top answer, and if it breaks a rule,
replace the main field with a fixed safe default.

The defaults were fixed before looking at the results: the "send it to a person / do nothing" option of
each scenario (leave -> hr_review, stock -> watch, shell -> ask). Uses the answers saved by bench/run.py,
so no model is called.

    python bench/fallback.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from run import broken, load_scenario  # noqa: E402

DEFAULTS = {"leave_request": {"verdict": "hr_review"}, "stock_alert": {"action": "watch"},
            "shell_guard": {"action": "ask"}}
SOURCES = {"leave_request": "kev-0.8b.json", "shell_guard": "kev-0.8b.json", "stock_alert": "stock-v2-kev.json"}


def main(defaults: dict = DEFAULTS) -> dict:
    tasks = {t["id"]: t for t in map(json.loads, (ROOT / "bench" / "tasks.jsonl").read_text().splitlines())}
    out = {}
    for scen, src in SOURCES.items():
        sc = load_scenario(scen)
        rows = [r for r in json.loads((ROOT / "bench" / "out" / src).read_text())["rows"]
                if tasks[r["id"]]["scenario"] == scen]
        c = {"n": 0, "model_only": 0, "fixed_default": 0, "shishya": 0, "default_used": 0, "default_breaks": 0}
        for r in rows:
            t = tasks[r["id"]]
            ctx, f = {"input": t["input"], "facts": t["facts"]}, t["field"]
            top, best = r["A"]["answer"], r["B"]["answer"] or {}
            fb = top
            if broken(sc, top, ctx):
                fb = {**top, **defaults[scen]}
                c["default_used"] += 1
                c["default_breaks"] += bool(broken(sc, fb, ctx))
            c["n"] += 1
            c["model_only"] += top[f] == t["gold"]
            c["fixed_default"] += fb[f] == t["gold"]
            c["shishya"] += best.get(f) == t["gold"]
        out[scen] = c
    return out


if __name__ == "__main__":
    res = main()
    print("| Scenario | Model only | Rule check + fixed default | Shishya | Default used |")
    print("|---|---:|---:|---:|---:|")
    tot = {k: 0 for k in ("n", "model_only", "fixed_default", "shishya", "default_used")}
    for scen, c in res.items():
        print(f"| {scen} ({c['n']}) | {c['model_only']} | {c['fixed_default']} | {c['shishya']} "
              f"| {c['default_used']} |")
        for k in tot:
            tot[k] += c[k]
    print(f"| **Total ({tot['n']})** | {tot['model_only']} | {tot['fixed_default']} | {tot['shishya']} | "
          f"{tot['default_used']} |")
