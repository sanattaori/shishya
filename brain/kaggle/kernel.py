"""Runs on Kaggle (GPU, internet on): train the three `bee` models, then run the life test.

brain/kaggle/run.sh uploads the code as a private dataset (the GitHub repo is private) and pushes this script.
Everything lands in /kaggle/working/out.
"""
import os
import shutil
import subprocess
import sys
import tarfile

CANARY = False          # run.sh --canary: 20 steps per model and a short life, to catch GPU-only bugs fast
W = "/kaggle/working"
OUT = f"{W}/out"
CODE = f"{W}/shishya"
CHILDHOOD = ["--preset", "bee", "--bytes", "1e8", "--batch", "4", "--accum", "4", "--seq", "256", "--segments",
             "4", "--lr", "1e-3", "--warmup", "200", "--log-every", "100", "--eval-every", "500",
             "--val-stories", "200", "--final-val-stories", "2000", "--data-dir", f"{W}/data"]


if CANARY:
    CHILDHOOD[CHILDHOOD.index("--bytes") + 1] = "3.3e5"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def unpack_code():
    """The dataset holds brain.tar.gz (Kaggle may have unpacked it already) and commit.txt."""
    os.makedirs(CODE, exist_ok=True)
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "brain.tar.gz" in files:
            with tarfile.open(os.path.join(root, "brain.tar.gz")) as t:
                t.extractall(CODE)
        elif os.path.basename(root) == "brain" and "models.py" in files and not os.path.exists(f"{CODE}/brain"):
            shutil.copytree(root, f"{CODE}/brain")
        if "commit.txt" in files:
            shutil.copy(os.path.join(root, "commit.txt"), f"{OUT}/commit.txt")
    assert os.path.exists(f"{CODE}/brain/models.py"), "code not found in /kaggle/input"


def train(kind: str, gpu: int):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT}/{kind}.log", "w")
    return subprocess.Popen([sys.executable, "-m", "brain.childhood", "--model", kind, *CHILDHOOD,
                             "--out", f"{OUT}/{kind}"], env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    unpack_code()
    os.chdir(CODE)
    sh([sys.executable, "-c", f"from brain import data; data.splits({W + '/data'!r}, train_bytes=400_000_000)"])
    import torch
    gpus = torch.cuda.device_count()
    print(f"{gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(gpus)]}", flush=True)
    if gpus >= 2:                                   # the Brain on one GPU, GPT then BDH on the other
        brain = train("brain", 1)
        for kind in ("gpt", "bdh"):
            if train(kind, 0).wait():
                raise SystemExit(f"{kind} failed: " + open(f"{OUT}/{kind}.log").read()[-2000:])
        if brain.wait():
            raise SystemExit("brain failed: " + open(f"{OUT}/brain.log").read()[-2000:])
    else:
        for kind in ("gpt", "bdh", "brain"):
            if train(kind, 0).wait():
                raise SystemExit(f"{kind} failed: " + open(f"{OUT}/{kind}.log").read()[-2000:])
    with open(f"{OUT}/life.log", "w") as log:
        r = subprocess.run([sys.executable, "-m", "brain.lifetest", "--runs", OUT, "--data-dir", f"{W}/data",
                            "--device", "cuda", "--out", f"{OUT}/life.json"] + (["--small"] if CANARY else []),
                           stdout=log, stderr=subprocess.STDOUT)
    if CANARY:
        for kind in ("gpt", "bdh", "brain"):
            print(f"== {kind}.log", open(f"{OUT}/{kind}.log").read()[-1500:], sep="\n")
    for kind in ("gpt", "bdh", "brain"):
        print(open(f"{OUT}/{kind}/metrics.json").read()[:400])
    print(open(f"{OUT}/life.log").read()[-4000:])
    if r.returncode:
        raise SystemExit("life test failed")


if __name__ == "__main__":
    main()
