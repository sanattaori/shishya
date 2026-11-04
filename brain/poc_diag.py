"""One-hour POC (docs/V8_POC.md): why a read fails. At the step that picks a name's first letter, which memory matches?

    python -m brain.poc_diag brain/out/poc/changes_dev.pt
"""
from __future__ import annotations

import sys

import torch

from .poc_read import Traces


def fact_names(d: dict) -> dict[int, tuple[int, str]]:
    """fact id → (index of its first trace, the name it spells)."""
    out: dict[int, list] = {}
    for i, (f, t) in enumerate(zip(d["fact"].tolist(), d["tokens"].tolist())):
        if f >= 0:
            out.setdefault(f, [i, bytearray()])[1].append(t)
    return {f: (v[0], bytes(v[1]).decode(errors="replace").strip()) for f, v in out.items()}


def main(path: str):
    d = torch.load(path, weights_only=False)
    tr = Traces(d)
    facts = fact_names(d)
    start_idx = torch.tensor([v[0] for v in facts.values()])
    names = [v[1] for v in facts.values()]
    print("facts", len(names), "sample names", names[:8])
    labels = ["plain", "other", "kept"] if len(d["exams"]) == 3 else ["plain", "other"]
    for lab, items in zip(labels, d["exams"]):
        top, gaps, rows = {"right": 0, "wrong option": 0, "other memory": 0, "none": 0}, [], []
        for q in range(0, len(items), 5):
            opts = [items[q + k]["cont"].strip(" .") for k in range(5)]
            it = items[q]
            s = tr.closeness(it["kidx"][1:2], it["kval"][1:2], it["what"][1:2])[0]        # the step after the space
            s0 = s[start_idx]
            b = int(s0.argmax())
            nm = names[b]
            if float(s0[b]) <= 0:
                top["none"] += 1
            elif nm == opts[0]:
                top["right"] += 1
            elif nm in opts:
                top["wrong option"] += 1
            else:
                top["other memory"] += 1
            best = {}
            for o in opts:
                v = [float(s0[i]) for i, n in enumerate(names) if n == o]
                best[o] = max(v) if v else None
            r = best[opts[0]]
            w = [x for o, x in best.items() if o != opts[0] and x is not None]
            if r is not None and w:
                gaps.append(r - max(w))
            if len(rows) < 6:
                rows.append((it["prompt"], [(o, None if best[o] is None else round(best[o], 3)) for o in opts], nm))
        print(f"\n== {lab}: the closest memory at the first letter, of {len(items) // 5} questions: {top}")
        if gaps:
            g = torch.tensor(gaps)
            print(f"right memory minus the best wrong option: mean {g.mean():.3f}, right is ahead on "
                  f"{float((g > 0).float().mean()):.0%}")
        for r in rows:
            print(" ", r)


if __name__ == "__main__":
    main(sys.argv[1])
