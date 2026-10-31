"""Can a question find its own fact in the lasting memory? The address probes behind the v2 amendment.

    python -m brain.probes model brain/out/tiny-long/brain --smoke     # a trained Brain's own addresses (v1 or v2)
    python -m brain.probes window --smoke                              # v2's fixed window code, untrained

For 100 made-up facts (dev names, never the test stream): the address at the step before the fact's name inside a
story, and the address at the same step of the bare question. A question "finds its fact" when its address is
closest to its own fact's address among all 100 (chance: 1%). Results are in brain/RESULTS.md.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from . import data
from .lifetest import load
from .models import Brain, BrainConfig


def fact_texts(sp: data.Splits, n: int = 100, seed: int = 9):
    """(fact, story text with the fact, byte position of the step before the name in the story)."""
    rng = random.Random(seed)
    names = sp.names["dev"][:]
    rng.shuffle(names)
    out = []
    for i in range(n):
        f = data.make_fact(rng, names[2 * i], names[2 * i + 1])
        text = data.with_facts(rng.choice(sp.life_dev), [f.sentence], data.SESSION_BYTES + 1)
        pos = text.encode().index(f.sentence.encode()) + len(f.prompt.encode())
        out.append((f, text, pos))
    return out


def match(store: torch.Tensor, query: torch.Tensor) -> dict:
    """store, query: [facts, heads, units], unit length per head. Similarity = mean over heads."""
    sim = torch.einsum("fhn,ghn->fg", query, store) / store.size(1)
    n = sim.size(0)
    return {"finds_own_fact": float((sim.argmax(1) == torch.arange(n)).float().mean()),
            "own_similarity": float(sim.diag().mean()),
            "other_similarity": float((sim.sum() - sim.diag().sum()) / (n * (n - 1)))}


def _dense(kidx: torch.Tensor, kval: torch.Tensor, units: int) -> torch.Tensor:
    return torch.zeros(units).scatter_(0, kidx, kval)


@torch.no_grad()
def model_addresses(model: Brain, text: str, pos: int) -> list[torch.Tensor]:
    """Per layer, the address the model would use at byte position ``pos``: [heads, units]."""
    x, _, _ = (torch.from_numpy(a)[None] for a in data.encode(text, data.SESSION_BYTES))
    model(x, keep=True)
    out = []
    for key, _ in model.trace:
        if model.cfg.key == "window":
            kidx, kval = key
            out.append(_dense(kidx[0, pos], kval[0, pos], model.cfg.key_size)[None])
        else:
            out.append(key[0, :, pos])
    model.trace = None
    return out


def probe_model(run: Path, sp: data.Splits) -> list[dict]:
    model = load(run, torch.device("cpu"))
    store, query = [], []
    for f, text, pos in fact_texts(sp):
        store.append(model_addresses(model, text, pos))
        query.append(model_addresses(model, f.prompt + " " + f.obj + ".", len(f.prompt.encode())))
    return [match(torch.stack([s[layer] for s in store]), torch.stack([q[layer] for q in query]))
            for layer in range(model.cfg.n_layer)]


def probe_window(sp: data.Splits, window: int, decay: float, units: int = 16384, active: int = 40) -> dict:
    cfg = BrainConfig(n_layer=1, n_embd=32, n_head=2, mult=8, mem_dim=8, key="window", key_size=units,
                      key_window=window, key_decay=decay, key_active=active)
    brain = Brain(cfg)
    store, query = [], []
    for f, text, pos in fact_texts(sp):
        q = f.prompt + " " + f.obj + "."
        for t, p, acc in ((text, pos, store), (q, len(f.prompt.encode()), query)):
            kidx, kval = brain.window_keys(torch.tensor([list(t.encode()[: data.SESSION_BYTES])]))
            acc.append(_dense(kidx[0, p], kval[0, p], units)[None])
    return match(torch.stack(store), torch.stack(query))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["model", "window"])
    p.add_argument("run", nargs="?", help="a Brain's run folder (for `model`)")
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    sp = data.splits(args.data_dir, smoke=args.smoke, need_train=False)
    if args.what == "model":
        for layer, r in enumerate(probe_model(Path(args.run), sp)):
            print(f"layer {layer}: finds its fact {r['finds_own_fact']:.0%} "
                  f"(own {r['own_similarity']:.2f} vs others {r['other_similarity']:.2f})")
        return
    for window in (16, 24, 32, 40, 48):
        for decay in (1.0, 0.95, 0.9):
            r = probe_window(sp, window, decay)
            print(f"window {window:2d} fading {decay}: finds its fact {r['finds_own_fact']:.0%} "
                  f"(own {r['own_similarity']:.2f} vs others {r['other_similarity']:.2f})")


if __name__ == "__main__":
    main()
