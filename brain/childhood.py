"""Childhood: train one model on TinyStories bytes plus memory drills. Every model sees the same bytes.

    python -m brain.childhood --model brain --preset tiny --smoke --bytes 3e6 --out brain/out/tiny/brain

An episode is ``--segments`` texts in a row. The Brain reads them in order and writes each one into its lasting
synapses before the next (backprop runs through those writes, so it learns how to use them). GPT and plain BDH
read the same texts one by one. Every model is scored on the same bytes: the mean loss over all of them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .models import build, config_of, masked_loss, n_params

LN2 = math.log(2)


def device_of(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def val_bpb(model, stories: list[str], T: int, device, batch: int = 32) -> float:
    """Bits per byte on the first T bytes of held-out stories, each read alone (no lasting memory)."""
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(stories), batch):
        enc = [data.encode(s, T) for s in stories[i:i + batch]]
        x, y, m = (torch.from_numpy(np.stack(a)).to(device) for a in zip(*enc))
        logits, _ = model(x)
        nll = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        tot += float((nll * m.flatten()).sum())
        cnt += int(m.sum())
    model.train()
    return tot / cnt / LN2


def save(model, args) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": args.model, "config": config_of(model), "state_dict": model.state_dict()}, out / "model.pt")
    return out


def train(args) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = device_of(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    sp = data.splits(args.data_dir, smoke=args.smoke, train_bytes=int(args.train_file_bytes))
    names = sp.names["train"]
    model = build(args.model, args.preset).to(dev)
    if getattr(model.cfg, "key_novelty", False):          # word counts for the v3 address, once, then fixed
        model.learn_familiarity(sp.train[:20000])
    lasting = getattr(model.cfg, "lasting", False)
    B, T, S = args.batch, args.seq, args.segments
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp = torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp)
    val = sp.val[: args.val_stories]
    A = args.accum
    steps = math.ceil(args.bytes / (A * B * S * T))
    mems = [model.new_memory(B, dev) for _ in range(A)] if lasting else None   # one per micro-batch slot
    hist, t0, seen = [], time.time(), 0
    print(f"{args.model}: {n_params(model):,} parameters, {steps} steps of {A}x{B}x{S}x{T} bytes on {dev}",
          flush=True)
    for step in range(steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for a in range(A):
            xs, ys, ms = (torch.from_numpy(v).to(dev)
                          for v in data.childhood_batch(rng, sp.train, names, B, T, S, args.p_drill))
            with amp:
                if lasting:
                    mem, nll_sum, n = mems[a], 0.0, ms.sum().clamp_min(1)
                    for s in range(S):
                        logits, loss_s = model(xs[:, s], ys[:, s], memory=mem, mask=ms[:, s], keep=s < S - 1)
                        nll_sum = nll_sum + loss_s * ms[:, s].sum()
                        if s < S - 1:
                            mem = model.write(mem, model.surprise(logits, ys[:, s], ms[:, s]), 1.0, ms[:, s])
                    loss = nll_sum / n
                    with torch.no_grad():                    # carry the memory on; reset a slot now and then,
                        keep_slot = (torch.rand(B, device=dev) > 1 / 64).float()
                        mems[a] = [m.detach() * keep_slot.view(B, *[1] * (m.dim() - 1))   # so memories have
                                   for m in mem]                                          # mixed ages
                    model.trace = None
                else:
                    logits, _ = model(xs.flatten(0, 1))
                    loss = masked_loss(logits, ys.flatten(0, 1), ms.flatten(0, 1))
            scaler.scale(loss / A).backward()
            step_loss += float(loss) / A
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        seen += A * B * S * T
        loss = torch.tensor(step_loss)
        if lasting:
            with torch.no_grad():
                model.loss_ref.mul_(0.99).add_(0.01 * step_loss)
        if step % args.log_every == 0 or step == steps - 1:
            row = {"step": step, "bytes": seen, "loss_bits": float(loss) / LN2, "lr": lr,
                   "seconds": round(time.time() - t0, 1)}
            if args.eval_every and (step % args.eval_every == 0 or step == steps - 1):
                row["val_bpb"] = val_bpb(model, val, T, dev)
                save(model, args)                     # so a stopped run keeps its latest model
            hist.append(row)
            print(json.dumps(row), flush=True)
    secs = time.time() - t0
    final = val_bpb(model, sp.val[: args.final_val_stories], T, dev)
    save(model, args)
    metrics = {"kind": args.model, "preset": args.preset, "params": n_params(model), "bytes": seen, "steps": steps,
               "val_bpb": final, "seconds": round(secs, 1), "bytes_per_s": round(seen / secs), "device": str(dev),
               "smoke": sp.smoke, "args": vars(args), "history": hist}
    if lasting:
        metrics["loss_ref"] = float(model.loss_ref)
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"done: val {final:.3f} bits/byte, {secs / 60:.1f} min, {seen / secs:,.0f} bytes/s", flush=True)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["gpt", "bdh", "brain", "brain2", "brain3", "brain4"], required=True)
    p.add_argument("--preset", choices=["tiny", "bee"], default="tiny")
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--smoke", action="store_true", help="train on part of the validation file (CPU check only)")
    p.add_argument("--train-file-bytes", type=float, default=4e8)
    p.add_argument("--bytes", type=float, default=3e6, help="training bytes in total")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=1, help="micro-batches per step (same bytes, less GPU memory)")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--p-drill", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--val-stories", type=int, default=200)
    p.add_argument("--final-val-stories", type=int, default=1000)
    p.add_argument("--out", required=True)
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
