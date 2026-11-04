"""POC (docs/V8_POC.md, step 2): reads that favour one memory per question. Dev sweep only, no reader runs.

    python -m brain.poc_follow changes.pt dense.pt --teach-changes t.json
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from . import poc_fix as X
from . import poc_table as T
from .poc_diag import fact_names


def spans(d: dict) -> tuple[torch.Tensor, list[list[int]]]:
    """Each fact's first trace, and all its traces in order."""
    by: dict[int, list[int]] = {}
    for i, f in enumerate(d["fact"].tolist()):
        if f >= 0:
            by.setdefault(f, []).append(i)
    keys = sorted(by)
    return torch.tensor([by[f][0] for f in keys]), [by[f] for f in keys]


def follow_score(rows, tb, tr, starts, facts, gain, table_w, hidden, sharp=None) -> list[bool]:
    """The closest live memory at the first letter is chosen for the question, and that memory spells the rest."""
    sc = torch.zeros(len(rows))
    for i, (r, b) in enumerate(zip(rows, tb)):
        lg = r["logits"].clone()
        if b is not None:
            lg[r["g"]] += table_w * b
            if sharp:
                lg[r["g"]] += T.R.bias(r["s"], tr, **sharp)
            g = r["g"].nonzero().flatten().tolist()
            s = r["s"]
            best = None
            for j, pos in enumerate(g):
                if pos != 1 and j != 0 and best is None:
                    continue
                if best is None:                      # the first gated step at or after the first letter
                    if pos < 1:
                        continue
                    v = s[j][starts].clone()
                    v[hidden] = -1
                    if float(v.max()) <= 0:
                        break
                    best, k, strength = int(v.argmax()), 0, float(v.max())
                ts = facts[best]
                if k < len(ts):
                    lg[pos] += gain * strength * (tr.Z[tr.tok[ts[k]]] @ tr.Z.T)
                    k += 1
        sc[i] = F.log_softmax(lg, -1).gather(1, r["tgt"][:, None]).sum()
    return (sc.view(-1, 5).argmax(1) == 0).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("changes")
    p.add_argument("dense")
    p.add_argument("--teach-changes", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(1)
    dc, trc, exc, tbc = T.load(a.changes, None)
    dd, trd, exd, tbd = T.load(a.dense, None)
    teach = json.load(open(a.teach_changes))
    vals = X.replaced_same_words(list(teach), teach, guard=True)
    X.hide(dc, exc, vals)
    names = fact_names(dc)
    ids = sorted(names)
    sc, fc = spans(dc)
    sd, fd = spans(dd)
    hc = torch.tensor([names[f][1] in vals for f in ids])
    hd = torch.zeros(len(sd), dtype=bool)
    m = T.R.mean
    rows = []
    for gain in (0, 4, 8, 16, 32):
        p_, o, k = (follow_score(e, b, trc, sc, fc, gain, 1.0, hc, X.SHARP) for e, b in zip(exc, tbc))
        dp, do = (follow_score(e, b, trd, sd, fd, gain, 1.0, hd, X.SHARP) for e, b in zip(exd, tbd))
        r = {"follow_gain": gain, "changes": m(p_ + o), "kept": m(k), "dense": m(dp), "dense_other": m(do)}
        r["dev_mean"] = (r["changes"] + r["kept"] + r["dense"]) / 3
        rows.append(r)
        print({k_: round(v, 3) for k_, v in r.items()}, flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
