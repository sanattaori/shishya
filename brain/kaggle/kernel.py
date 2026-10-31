"""Runs on Kaggle (GPU, internet on): clone the branch, train the three `bee` models, run the life test.

Pushed by brain/kaggle/run.sh, which writes the branch name in below. Everything lands in /kaggle/working/out.
"""
import os
import subprocess
import sys

REPO = "https://github.com/sanattaori/shishya"
BRANCH = "BRANCH_NAME"
W = "/kaggle/working"
OUT = f"{W}/out"
CHILDHOOD = ["--preset", "bee", "--bytes", "1e8", "--batch", "4", "--accum", "4", "--seq", "256", "--segments",
             "4", "--lr", "1e-3", "--warmup", "200", "--log-every", "100", "--eval-every", "500",
             "--val-stories", "200", "--final-val-stories", "2000", "--data-dir", f"{W}/data"]


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def train(kind: str, gpu: int):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT}/{kind}.log", "w")
    return subprocess.Popen([sys.executable, "-m", "brain.childhood", "--model", kind, *CHILDHOOD,
                             "--out", f"{OUT}/{kind}"], env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, f"{W}/shishya"])
    os.chdir(f"{W}/shishya")
    os.makedirs(OUT, exist_ok=True)
    sh(["git", "rev-parse", "HEAD"], stdout=open(f"{OUT}/commit.txt", "w"))
    sh([sys.executable, "-c", f"from brain import data; data.splits({W + '/data'!r}, train_bytes=400_000_000)"])
    import torch
    gpus = torch.cuda.device_count()
    print(f"{gpus} GPU(s)", flush=True)
    if gpus >= 2:                                   # the Brain on one GPU, GPT then BDH on the other
        brain = train("brain", 1)
        for kind in ("gpt", "bdh"):
            if train(kind, 0).wait():
                raise SystemExit(f"{kind} failed")
        if brain.wait():
            raise SystemExit("brain failed")
    else:
        for kind in ("gpt", "bdh", "brain"):
            if train(kind, 0).wait():
                raise SystemExit(f"{kind} failed")
    with open(f"{OUT}/life.log", "w") as log:
        sh([sys.executable, "-m", "brain.lifetest", "--runs", OUT, "--data-dir", f"{W}/data", "--device", "cuda",
            "--out", f"{OUT}/life.json"], stdout=log, stderr=subprocess.STDOUT)
    for kind in ("gpt", "bdh", "brain"):            # keep the output small: metrics and logs, plus the models
        print(open(f"{OUT}/{kind}/metrics.json").read()[:400])
    print(open(f"{OUT}/life.log").read()[-3000:])


if __name__ == "__main__":
    main()
