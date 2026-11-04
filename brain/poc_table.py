"""One-hour POC (docs/V8_POC.md): v7's table rebuilt from saved traces, alone, with a role map, and with a sharp read.

The rebuild replays the exact sequential delta rule, so with no map it must give v7's own scores (the control).

    python -m brain.poc_table brain/out/poc/changes_dev.pt brain/out/poc/dense_dev.pt [--map role_map.pt]
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from . import poc_read as R
from . import v5


class _Reader:
    def __init__(self, m: int):
        self.m = m


class Table:
    """The delta rule's memory, as in v5.Student, built from traces."""

    def __init__(self, d: dict, m: int = 128):
        self.reader = _Reader(m)
        M = torch.zeros(16384, d["Wh"].size(1), m)
        K, V, W = d["K_i"].long(), d["K_v"].float(), d["Wh"].float()
        Z, tok, beta = d["Z"].float(), d["tokens"], d["beta"]
        for c in range(0, len(tok), 128):
            ix = slice(c, c + 128)
            M = v5.write(M, K[ix], V[ix], W[ix], Z[tok[ix]], beta[ix], self)
        self.M = M

    read = v5.Student.read


def table_bias(rows: list[dict], items: list[dict], t: Table, Z: torch.Tensor, gain: float) -> list:
    out = []
    for r, it in zip(rows, items):
        g = r["g"]
        out.append(gain * (t.read(it["kidx"][g].long(), it["kval"][g].float(), it["what"][g].float()) @ Z.T)
                   if g.any() else None)
    return out


def score(rows, tb, tr, weight: float, kw: dict | None) -> list[bool]:
    sc = torch.zeros(len(rows))
    for i, (r, b) in enumerate(zip(rows, tb)):
        lg = r["logits"].clone()
        if b is not None:
            lg[r["g"]] += weight * b
            if kw is not None:
                lg[r["g"]] += R.bias(r["s"], tr, **kw)
        sc[i] = F.log_softmax(lg, -1).gather(1, r["tgt"][:, None]).sum()
    return (sc.view(-1, 5).argmax(1) == 0).tolist()


def load(path: str, A) -> tuple[dict, R.Traces, list, list]:
    d = torch.load(path, weights_only=False)
    if A is not None:
        d["Wh"] = F.normalize(d["Wh"].float() @ A, dim=-1)
        for items in d["exams"]:
            for it in items:
                it["what"] = F.normalize(it["what"].float() @ A, dim=-1)
    tr, ex = R.prepare(d)
    t = Table(d)
    tb = [table_bias(rows, items, t, tr.Z, d["gain"]) for rows, items in zip(ex, d["exams"])]
    return d, tr, ex, tb


SETTINGS = {"table": (1.0, None)}
for g in (2, 4, 8):
    for p, n in ((8, 0.8), (8, 0.9), (4, 0.9)):
        for newest in (True, False):
            SETTINGS[f"table+sharp g{g} p{p} n{n} {'newest' if newest else 'sum'}"] = (
                1.0, dict(power=p, near=n, gain=g, newest=newest))


def evaluate(dc, trc, exc, tbc, dd, trd, exd, tbd, weight, kw) -> dict:
    p, o, k = (score(e, b, trc, weight, kw) for e, b in zip(exc, tbc))
    dp, do = (score(e, b, trd, weight, kw) for e, b in zip(exd, tbd))
    m = R.mean
    return {"changes": m(p + o), "plain": m(p), "other": m(o), "kept": m(k), "dense": m(dp), "dense_other": m(do),
            "dev_mean": (m(p + o) + m(k) + m(dp)) / 3, "correct": {"cur": p + o, "kept": k, "dp": dp, "do": do}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("changes")
    p.add_argument("dense")
    p.add_argument("--map")
    p.add_argument("--only", help="one setting's name, for the test")
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(2)
    A = torch.load(a.map, weights_only=False)["A"] if a.map else None
    dc, trc, exc, tbc = load(a.changes, A)
    dd, trd, exd, tbd = load(a.dense, A)
    print("v7 as collected:", {k: v for k, v in dc["res"].items() if k[:4] in ("curr", "kept")},
          {k: v for k, v in dd["res"].items() if k in ("plain", "other_words")}, flush=True)
    rows = {}
    for name, (w, kw) in SETTINGS.items():
        if a.only and name != a.only:
            continue
        r = evaluate(dc, trc, exc, tbc, dd, trd, exd, tbd, w, kw)
        rows[name] = r
        print(f"{r['dev_mean']:.3f} {name}: changes {r['changes']:.3f} kept {r['kept']:.3f} one sentence "
              f"{r['dense']:.3f}/{r['dense_other']:.3f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
