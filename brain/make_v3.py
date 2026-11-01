"""Make the v3 Brain: v2's trained network, with v3's address and write rule (``models.V3``).

    python -m brain.make_v3 brain/out/runs/brain2 brain/out/runs/brain3

v3 changes only how the lasting memory is addressed and written, so the network itself (BDH core, read path,
learned write rates) is v2's, as trained in childhood. The fixed random codes are made again from their seed. With
``key_novelty`` the word counts come from the first 20,000 childhood stories, as in ``brain.study``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import data
from .models import V3, from_config


def make(src: Path, dst: Path, data_dir: str) -> None:
    ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    assert ck["kind"] == "brain2", f"v3 is made from a v2 Brain, not {ck['kind']}"
    cfg = {**ck["config"], **V3}
    brain = from_config("brain3", cfg)
    state = {k: v for k, v in ck["state_dict"].items() if k not in ("key_byte", "key_pos")}   # made from the seed
    missing, unexpected = brain.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"key_byte", "key_pos", "key_count"}, (missing, unexpected)
    if cfg.get("key_novelty"):
        brain.learn_familiarity(data.splits(data_dir, need_train=False).train[:20000])
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "brain3", "config": cfg, "state_dict": brain.state_dict()}, dst / "model.pt")
    metrics = json.loads((src / "metrics.json").read_text())
    metrics.update(kind="brain3", made_from=str(src), v3_settings=V3)
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"v3 Brain written to {dst}: v2's network with {V3}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="the v2 Brain's run folder")
    p.add_argument("dst")
    p.add_argument("--data-dir", default="brain/out/data")
    args = p.parse_args(argv)
    make(Path(args.src), Path(args.dst), args.data_dir)


if __name__ == "__main__":
    main()
