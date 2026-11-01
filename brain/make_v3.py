"""Make the v3 (or v4) Brain: v2's trained network, with that version's memory settings (``models.VERSIONS``).

    python -m brain.make_v3 brain/out/runs/brain2 brain/out/runs/brain3
    python -m brain.make_v3 brain/out/runs/brain2 brain/out/runs/brain4 --kind brain4

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
from .models import VERSIONS, from_config


def make(src: Path, dst: Path, data_dir: str, kind: str = "brain3") -> None:
    ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    assert ck["kind"] == "brain2", f"{kind} is made from a v2 Brain, not {ck['kind']}"
    settings = VERSIONS[kind]
    cfg = {**ck["config"], **settings}
    brain = from_config(kind, cfg)
    state = {k: v for k, v in ck["state_dict"].items() if k not in ("key_byte", "key_pos")}   # made from the seed
    missing, unexpected = brain.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"key_byte", "key_pos", "key_count"}, (missing, unexpected)
    if cfg.get("key_novelty"):
        brain.learn_familiarity(data.splits(data_dir, need_train=False).train[:20000])
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "config": cfg, "state_dict": brain.state_dict()}, dst / "model.pt")
    metrics = json.loads((src / "metrics.json").read_text())
    metrics.update(kind=kind, made_from=str(src), settings=settings)
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"{kind} written to {dst}: v2's network with {settings}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="the v2 Brain's run folder")
    p.add_argument("dst")
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--kind", default="brain3", choices=sorted(VERSIONS))
    args = p.parse_args(argv)
    make(Path(args.src), Path(args.dst), args.data_dir, args.kind)


if __name__ == "__main__":
    main()
