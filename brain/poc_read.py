"""One-hour POC (docs/V8_POC.md): try read designs over the data saved by ``brain.poc_collect``. No reader runs.

The design: no big table. Each memory is kept apart, as v7's write trace. A question finds its closest traces, how close
being (who overlap) × (what cosine), as in v7. Three changes from v7's table:

- **Sharp:** closeness counts as s ** power, so a close memory beats many half-close ones.
- **Newest wins:** among traces within ``near`` of the best match, the newest one answers.
- **Strength:** the answer is as strong as the best match, so an unknown address changes nothing.

    python -m brain.poc_read sweep brain/out/poc/changes_dev.pt brain/out/poc/dense_dev.pt
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch
import torch.nn.functional as F


def who_rows(kidx: torch.Tensor, kval: torch.Tensor, U: int = 16384) -> torch.Tensor:
    return torch.zeros(kidx.size(0), U).scatter_(-1, kidx.long(), kval.float())


class Traces:
    def __init__(self, d: dict):
        self.K = who_rows(d["K_i"], d["K_v"])
        self.W = d["Wh"].float()
        self.tok = d["tokens"]
        self.Z = d["Z"].float()
        self.gain = d["gain"]

    def closeness(self, kidx, kval, what) -> torch.Tensor:
        q = who_rows(kidx, kval)
        return (q @ self.K.T) * (what.float() @ self.W.T)


def bias(s: torch.Tensor, tr: Traces, power: float, near: float, gain: float, newest: bool) -> torch.Tensor:
    """s: [steps, traces] → logit bias [steps, tokens]."""
    out = torch.zeros(s.size(0), tr.Z.size(0))
    n = s.size(1)
    for q in range(s.size(0)):
        best = float(s[q].max())
        if best <= 0:
            continue
        keep = (s[q] >= best * near).nonzero().flatten()
        if newest:
            out[q] = (best ** power) * (tr.Z[tr.tok[keep[-1]]] @ tr.Z.T)
        else:
            w = s[q, keep].clamp(min=0) ** power
            out[q] = (w[:, None] * tr.Z[tr.tok[keep]]).sum(0) @ tr.Z.T
    del n
    return gain * out


def prepare(d: dict) -> tuple[Traces, list[dict]]:
    """Each exam's options, with every option's closeness matrix worked out once."""
    tr = Traces(d)
    exams = []
    for items in d["exams"]:
        rows = []
        for it in items:
            g = it["gate"]
            s = tr.closeness(it["kidx"][g], it["kval"][g], it["what"][g]) if g.any() else None
            rows.append({"logits": it["logits"].float(), "tgt": it["tgt"].long(), "g": g, "s": s})
        exams.append(rows)
    return tr, exams


def score(rows: list[dict], tr: Traces, power: float, near: float, gain: float, newest: bool) -> list[bool]:
    sc = torch.zeros(len(rows))
    for i, r in enumerate(rows):
        lg = r["logits"].clone()
        if r["s"] is not None:
            lg[r["g"]] += bias(r["s"], tr, power, near, gain, newest)
        sc[i] = F.log_softmax(lg, -1).gather(1, r["tgt"][:, None]).sum()
    return (sc.view(-1, 5).argmax(1) == 0).tolist()


def mean(x: list[bool]) -> float:
    return sum(x) / max(1, len(x))


def run_changes(tr, exams, **kw) -> dict:
    p, o, k = (score(e, tr, **kw) for e in exams)
    return {"plain": mean(p), "other": mean(o), "both": mean(p + o), "kept": mean(k), "correct": p + o + k}


def run_dense(tr, exams, **kw) -> dict:
    p, o = (score(e, tr, **kw) for e in exams)
    return {"plain": mean(p), "other": mean(o), "correct": p + o}


GRID = {"power": [1, 2, 4, 8], "near": [1.0, 0.98, 0.95, 0.9, 0.8], "gain": [8, 16, 32], "newest": [True, False]}


def dev_mean(a: dict, b: dict) -> float:
    """The dev rule, fixed before the sweep: the plain mean of v7's three hard scores (Y1, Y2, Y3 plain)."""
    return (a["both"] + a["kept"] + b["plain"]) / 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sweep", "one"])
    p.add_argument("changes")
    p.add_argument("dense")
    p.add_argument("--power", type=float, default=4)
    p.add_argument("--near", type=float, default=0.95)
    p.add_argument("--gain", type=float, default=16)
    p.add_argument("--newest", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--map", help="a role map from brain.poc_roles, applied to every what vector")
    a = p.parse_args()
    torch.set_num_threads(2)
    dc, dd = torch.load(a.changes, weights_only=False), torch.load(a.dense, weights_only=False)
    if a.map:
        A = torch.load(a.map, weights_only=False)["A"]
        for d in (dc, dd):
            d["Wh"] = F.normalize(d["Wh"].float() @ A, dim=-1)
            for items in d["exams"]:
                for it in items:
                    it["what"] = F.normalize(it["what"].float() @ A, dim=-1)
    print("control, v7's table on these questions:", {k: v for k, v in dc["res"].items() if k[:4] in ("curr", "kept")},
          {k: v for k, v in dd["res"].items() if k in ("plain", "other_words")}, flush=True)
    trc, exc = prepare(dc)
    trd, exd = prepare(dd)
    if a.mode == "one":
        kw = dict(power=a.power, near=a.near, gain=a.gain, newest=bool(a.newest))
        rc, rd = run_changes(trc, exc, **kw), run_dense(trd, exd, **kw)
        out = {"settings": kw, "changes": rc, "dense": rd}
        print(json.dumps({"settings": kw, "changes": {k: v for k, v in rc.items() if k != "correct"},
                          "dense": {k: v for k, v in rd.items() if k != "correct"}}))
        if a.out:
            json.dump(out, open(a.out, "w"))
        return
    rows = []
    for power, near, gain, newest in itertools.product(*GRID.values()):
        if not newest and near == 1.0:
            continue
        kw = dict(power=power, near=near, gain=gain, newest=newest)
        rc, rd = run_changes(trc, exc, **kw), run_dense(trd, exd, **kw)
        rows.append((dev_mean(rc, rd), kw, rc["both"], rc["kept"], rd["plain"], rd["other"]))
        print(f"{rows[-1][0]:.3f} {kw} changes {rc['both']:.3f} kept {rc['kept']:.3f} dense {rd['plain']:.3f}/"
              f"{rd['other']:.3f}", flush=True)
    rows.sort(key=lambda r: -r[0])
    print("BEST", rows[:5])
    if a.out:
        json.dump([list(r) for r in rows], open(a.out, "w"))


if __name__ == "__main__":
    main()
