"""One-hour POC (docs/V8_POC.md): a small map on the "what" vector, so wordings of one kind of fact sit together.

Trained once, with gradients, on the childhood relations only, with train names. The tested relations are never seen.

    python -m brain.poc_roles collect --out brain/out/poc/role_data.pt
    python -m brain.poc_roles train --data brain/out/poc/role_data.pt --out brain/out/poc/role_map.pt
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from . import v5, v6
from .poc_collect import BDH, RUN


def wordings(rel: str, T: dict) -> list[str]:
    """Each with {s} and {o}, the value last. Lessons, changes, and clauses after the person's name."""
    out = [w for w in v6.lesson_wordings(rel, T) if w.rstrip(". ").endswith("{o}")][:5]
    out += [w for w in T[rel]["change"] if w.rstrip(". ").endswith("{o}")][:3]
    out += ["{s} " + c + "." for c in T[rel]["clause"] if c.rstrip(". ").endswith("{o}")][:2]
    return out


@torch.no_grad()
def collect(args):
    torch.set_num_threads(args.threads)
    st, sp, _, _ = v5.load_student(RUN, BDH, "brain/out/data", "cpu")
    T = v6.teacher()
    rng = random.Random(5)
    names = sp.names["train"]
    texts, labels, rels = [], [], sorted(v5.RELATIONS)         # never the five tested relations
    for i, r in enumerate(rels):
        for w in wordings(r, T):
            for _ in range(args.names):
                s, o = rng.sample(names, 2)
                texts.append((w.format(s=s, o=o), o))
                labels.append(i)
    print(len(texts), "texts", len(rels), "relations", flush=True)
    whats, keep = [], []
    for j in range(0, len(texts), 32):
        chunk = texts[j:j + 32]
        ps = v5.Parsed(st.tok, st.who, [t for t, _ in chunk])
        st.meaning(ps)
        for n, (t, o) in enumerate(chunk):
            c = t.rindex(o)                                 # the step whose next token is the value's first letter
            k = next((k for k, x in enumerate(ps.tk.char_starts[n]) if k > 0 and x >= c), None)
            good = k is not None and bool(ps.gate[n, k - 1])
            keep.append(good)
            whats.append(ps.what[n, k - 1].clone() if good else torch.zeros(st.reader.r))
        print(j, flush=True)
    keep = torch.tensor(keep)
    torch.save({"what": torch.stack(whats)[keep], "label": torch.tensor(labels)[keep], "rels": rels,
                "dev": [rels.index(r) for r in v5.DEV_RELATIONS]}, args.out)
    print("DONE kept", int(keep.sum()), "of", len(keep), flush=True)


def apply_map(A: torch.Tensor, what: torch.Tensor) -> torch.Tensor:
    return F.normalize(what.float() @ A, dim=-1)


def same_kind(x: torch.Tensor, y: torch.Tensor) -> float:
    """For each wording, is its nearest other vector the same kind of fact?"""
    s = x @ x.T
    s.fill_diagonal_(-2)
    return float((y[s.argmax(1)] == y).float().mean())


def train(args):
    d = torch.load(args.data, weights_only=False)
    x, y = d["what"].float(), d["label"]
    dev = torch.tensor([bool(int(v) in d["dev"]) for v in y])
    A = torch.nn.Parameter(torch.eye(x.size(1)))
    opt = torch.optim.Adam([A], lr=args.lr)
    xt, yt = x[~dev], y[~dev]
    for step in range(args.steps):
        z = apply_map(A, xt)
        sim = z @ z.T / args.temp
        sim = sim - 1e4 * torch.eye(len(z))
        pos = (yt[None] == yt[:, None]).float() - torch.eye(len(z))
        lp = sim - sim.logsumexp(1, keepdim=True)
        loss = -((lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean() + args.keep * (A - torch.eye(len(A))).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                print(step, f"loss {float(loss):.3f} same kind: train {same_kind(apply_map(A, xt), yt):.2f}, "
                      f"held-out relations {same_kind(apply_map(A, x[dev]), y[dev]):.2f} "
                      f"(no map {same_kind(F.normalize(x[dev], dim=-1), y[dev]):.2f})", flush=True)
    torch.save({"A": A.detach(), "args": vars(args)}, args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["collect", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--data")
    p.add_argument("--names", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--keep", type=float, default=0.01)
    a = p.parse_args()
    (collect if a.mode == "collect" else train)(a)


if __name__ == "__main__":
    main()
