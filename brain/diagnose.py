"""Does the Brain use its lasting memory at all? A diagnostic, not a pass rule.

    python -m brain.diagnose brain/out/tiny --smoke

Childhood-style drills with dev names: facts in the first text, two other stories, then the facts again. We measure
the loss on the made-up object names in the last text, for the Brain with the memory it wrote and with an empty
memory, and for GPT and BDH (they have none). If memory helps here but the life test fails, the memory works for
short gaps and the problem is the long life; if it doesn't help here, the memory never learned to carry facts.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .childhood import LN2, device_of
from .lifetest import load


def drills(sp: data.Splits, n: int, T: int, seed: int = 5):
    """n episodes of 4 texts, with the byte spans of the object names in the last text."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        k = rng.randint(1, 3)
        picked = rng.sample(sp.names["dev"], 2 * k)
        facts = [data.make_fact(rng, picked[2 * i], picked[2 * i + 1]) for i in range(k)]
        first = data.with_facts(rng.choice(sp.life_dev), [f.sentence for f in facts], T + 1)
        mid = [rng.choice(sp.life_dev) for _ in range(2)]
        rng.shuffle(facts)
        last = " ".join(f.sentence for f in facts)
        spans, pos = [], 0
        for f in facts:
            start = pos + len(f.sentence.encode()) - len(f.obj.encode()) - 1   # the object, before the full stop
            spans.append((start, start + len(f.obj.encode())))
            pos += len(f.sentence.encode()) + 1
        out.append(([first] + mid + [data._fit(last + " " + rng.choice(sp.life_dev), T + 1)], spans))
    return out


@torch.no_grad()
def object_bits(model, episodes, T: int, device, use_memory: bool) -> float:
    """Mean bits per byte on the object names in the last text."""
    lasting = getattr(model.cfg, "lasting", False) and use_memory
    tot, cnt = 0.0, 0
    for texts, spans in episodes:
        mem = model.new_memory(1, device) if lasting else None
        for s, text in enumerate(texts):
            x, y, m = (torch.from_numpy(a)[None].to(device) for a in data.encode(text, T))
            logits, _ = model(x, memory=mem, keep=lasting and s < 3)
            if lasting and s < 3:
                mem = model.write(mem, model.surprise(logits, y, m), 1.0, m)
        nll = F.cross_entropy(logits[0].float(), y[0], reduction="none")
        for a, b in spans:                     # byte j is predicted at position j - 1
            tot += float(nll[a - 1:b - 1].sum())
            cnt += b - a
    return tot / cnt / LN2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs")
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    dev = device_of(args.device)
    sp = data.splits(args.data_dir, smoke=args.smoke, need_train=False)
    eps = drills(sp, args.n, data.SESSION_BYTES)
    res = {}
    for kind in ("gpt", "bdh", "brain"):
        model = load(Path(args.runs) / kind, dev)
        res[kind] = object_bits(model, eps, data.SESSION_BYTES, dev, use_memory=True)
        if kind == "brain":
            res["brain_empty_memory"] = object_bits(model, eps, data.SESSION_BYTES, dev, use_memory=False)
    res["memory_gain_bits"] = res["brain_empty_memory"] - res["brain"]
    print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
    return res


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
