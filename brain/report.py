"""Turn a life-test run into the tables of brain/RESULTS.md.

    python -m brain.report brain/out/kaggle-shishya-brain-life/out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {"gpt": "GPT + fine-tuning", "bdh": "BDH + fine-tuning", "brain": "**Shishya Brain** (local rule only)"}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def report(run: Path) -> str:
    life = json.loads((run / "life.json").read_text())
    brain_dir = life.get("setup", {}).get("brain", "brain")          # "brain2" for the v2 run
    met = {k: json.loads((run / (brain_dir if k == "brain" else k) / "metrics.json").read_text())
           for k in ("gpt", "bdh", "brain")}
    out = ["### Childhood (same bytes for every model)", "",
           "| Model | Parameters | Bytes | Held-out bits/byte | Hours | Bytes/s |", "|---|---:|---:|---:|---:|---:|"]
    for k, m in met.items():
        out.append(f"| {k} | {m['params']:,} | {m['bytes']:,} | {m['val_bpb']:.3f} | {m['seconds'] / 3600:.1f} | "
                   f"{m['bytes_per_s']:,} |")
    g = met["gpt"]["val_bpb"]
    out += ["", f"Stage 1 (within 5% of GPT): BDH {met['bdh']['val_bpb'] / g - 1:+.1%}, "
            f"Brain {met['brain']['val_bpb'] / g - 1:+.1%}.", "",
            "### Life (test stream, fresh-context questions, chance = 20%)", "",
            "| Learner | Setting (dev) | Recall, 3 exposures | 1 exposure | 1 correction (reward 3) "
            "| Changed fact: new wins | Paraphrased | Early facts kept | Bits/byte change |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, t in life["test"].items():
        out.append(f"| {NAMES[k]} | {life['chosen'][k]} | {pct(t['recall_main'])} | {pct(t['recall_once'])} | "
                   f"{pct(t['recall_reward'])} | {pct(t['change_new_wins'])} | {pct(t['recall_paraphrase'])} | "
                   f"{t['early_end']:.2f} of {t['early_mid']:.2f} | {t['bpb_rise']:+.1%} |")
    out.append("| No learning (as after childhood) | | " + " / ".join(
        f"{k} {pct(v)}" for k, v in life["no_learning"].items()) + " | | | | | | |")
    v, r = life["verdict"], life["resources"]
    out += ["", "### Verdict (rules fixed in docs/PLAN.md before the run)", "", "| Rule | Result |", "|---|---|"]
    for name, ok in v["rules"].items():
        out.append(f"| {name} | {'pass' if ok else '**fail**'} |")
    l3 = ", ".join(f"vs {k}: recall difference lower 95% bound {d['recall_diff_lower95']:+.2f}"
                   for k, d in v["L3_detail"].items())
    out += ["", "L3 detail: " + l3,
            f"L4 measured: {r['ms_per_write']:.0f} ms per 256-byte write, peak {r['peak_rss_mb']:.0f} MB.", "",
            f"**Overall: {'PASS' if v['pass'] else 'FAIL'}**"]
    return "\n".join(out)


if __name__ == "__main__":
    print(report(Path(sys.argv[1])))
