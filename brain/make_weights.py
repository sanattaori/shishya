"""Pack what `python -m brain.teach` needs into brain/weights/, from a finished v7 run. Only the maintainer needs this.

    python -m brain.make_weights brain/out/kaggle-shishya-brain-v7/out/v7/l3

Writes the reader v7's life test used (childhood, 500 steps, layer 3) and the word counts behind its address: how
often each hashed word occurs in the first 20,000 childhood stories. The counts are stored sparsely, and the check at
the end says they rebuild the address exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from . import data
from .hybrid import make_address
from .teach import ADDRESS, READER, load_address


def main() -> None:
    run = Path(sys.argv[1])
    ck = torch.load(run / "reader.pt", weights_only=False)
    a = {k: ck["args"][k] for k in ("who", "pool", "layer", "r", "power")}
    READER.parent.mkdir(exist_ok=True)
    torch.save({"reader": ck["reader"], "args": a, "loss_ref": ck["loss_ref"]}, READER)
    base = make_address(0.0, data.splits("brain/out/data", need_train=False).train[:20000])
    c = base.key_count
    i = c.nonzero().flatten()
    torch.save({"index": i.int(), "count": c[i].int()}, ADDRESS)
    back = load_address()
    for k, v in base.state_dict().items():
        if k.startswith("key"):
            assert torch.equal(v, back.state_dict()[k]), k
    kb = READER.stat().st_size / 1e3, ADDRESS.stat().st_size / 1e3
    print(f"reader {kb[0]:.0f} KB, address {kb[1]:.0f} KB: rebuilt exactly")


if __name__ == "__main__":
    main()
