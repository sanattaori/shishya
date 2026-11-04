"""POC (docs/V8_POC.md): why one sentence fails. Which option wins, by kind of fact, and which memory is closest.

    python -m brain.poc_dense_diag guru_dense.pt
"""
from __future__ import annotations

import collections
import sys

import torch
import torch.nn.functional as F

from . import poc_table as T
from .poc_diag import fact_names
from .poc_fix import SHARP


def kind(prompt: str) -> str:
    for w, k in (("town", "town"), ("toy", "toy"), ("friend", "friend"), ("secret", "secret"), ("word", "secret"),
                 ("called", "pet"), ("belongs", "pet")):
        if w in prompt:
            return k
    return "?"


def main(path: str):
    torch.set_num_threads(1)
    d, tr, ex, tb = T.load(path, None)
    names = fact_names(d)
    start = torch.tensor([v[0] for v in names.values()])
    nm = [v[1] for v in names.values()]
    for lab, rows, items, b in zip(("plain", "other"), ex, d["exams"], tb):
        for mode, kw in (("table", None), ("table+sharp", SHARP)):
            tally = collections.Counter()
            rel = collections.defaultdict(lambda: [0, 0])
            for q in range(0, len(rows), 5):
                sc = []
                for i in range(q, q + 5):
                    r = rows[i]
                    lg = r["logits"].clone()
                    if b[i] is not None:
                        lg[r["g"]] += b[i]
                        if kw:
                            lg[r["g"]] += T.R.bias(r["s"], tr, **kw)
                    sc.append(float(F.log_softmax(lg, -1).gather(1, r["tgt"][:, None]).sum()))
                w = max(range(5), key=lambda k: sc[k])
                k = kind(items[q]["prompt"])
                rel[k][0] += w == 0
                rel[k][1] += 1
                tally["right" if w == 0 else "own other fact" if w in (1, 2) else "outsider"] += 1
            print(lab, mode, dict(tally), {k: f"{a}/{n}" for k, (a, n) in sorted(rel.items())}, flush=True)
        close = collections.Counter()
        cos = []
        for q in range(0, len(rows), 5):
            it = items[q]
            opts = [items[q + k]["cont"].strip(" .") for k in range(5)]
            s = tr.closeness(it["kidx"][1:2], it["kval"][1:2], it["what"][1:2])[0][start]
            b0 = nm[int(s.argmax())]
            close["right" if b0 == opts[0] else "own other" if b0 in opts[1:3] else "other"] += 1
            own = [(float(s[j]), n) for j, n in enumerate(nm) if n in opts[:3]]
            r0 = [x for x, n in own if n == opts[0]]
            w0 = [x for x, n in own if n != opts[0]]
            if r0 and w0:
                cos.append((r0[0], max(w0)))
        c = torch.tensor(cos)
        print(lab, "closest memory at the first letter:", dict(close), f"closeness right {c[:, 0].mean():.3f}, "
              f"best own wrong {c[:, 1].mean():.3f}, right ahead {float((c[:, 0] > c[:, 1]).float().mean()):.0%}")
    W = F.normalize(d["Wh"].float()[start], dim=-1)
    sim = (W @ W.T)
    off = sim[~torch.eye(len(W), dtype=bool)]
    print("what cosine between stored facts: mean", round(float(off.mean()), 3), "min", round(float(off.min()), 3))


if __name__ == "__main__":
    main(sys.argv[1])
