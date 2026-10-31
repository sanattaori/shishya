"""Runs on Kaggle (GPU, internet on). brain/kaggle/run.sh uploads the code as a private dataset (the GitHub repo
is private), writes MODE below, and pushes this script. Everything lands in /kaggle/working/out.

- full:   train GPT, BDH and the v1 Brain on the same 100M bytes, then the life test
- canary: the same with 20 steps per model and a short life, to catch GPU-only bugs in minutes
- brain2: train only the v2 Brain (same recipe and bytes)
- life2:  the life test of the v2 Brain against the GPT and BDH of the `full` run (both runs' outputs mounted)
"""
import os
import shutil
import subprocess
import sys
import tarfile

MODE = "full"
W = "/kaggle/working"
OUT = f"{W}/out"
CODE = f"{W}/shishya"
CHILDHOOD = ["--preset", "bee", "--bytes", "1e8", "--batch", "4", "--accum", "4", "--seq", "256", "--segments",
             "4", "--lr", "1e-3", "--warmup", "200", "--log-every", "100", "--eval-every", "500",
             "--val-stories", "200", "--final-val-stories", "2000", "--data-dir", f"{W}/data"]
if MODE == "canary":
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
            if "commit.txt" in files:              # only the code dataset's; mounted runs carry their own
                shutil.copy(os.path.join(root, "commit.txt"), f"{OUT}/commit.txt")
        elif os.path.basename(root) == "brain" and "models.py" in files and not os.path.exists(f"{CODE}/brain"):
            shutil.copytree(root, f"{CODE}/brain")
    assert os.path.exists(f"{CODE}/brain/models.py"), "code not found in /kaggle/input"


def mount_runs(names):
    """Copy earlier kernels' trained models (folders with model.pt) from /kaggle/input into OUT."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        name = os.path.basename(root)
        if name in names and "model.pt" in files and not os.path.exists(f"{OUT}/{name}"):
            shutil.copytree(root, f"{OUT}/{name}")
    missing = [n for n in names if not os.path.exists(f"{OUT}/{n}/model.pt")]
    assert not missing, f"not found in /kaggle/input: {missing}"


def train(kind: str, gpu: int):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT}/{kind}.log", "w")
    return subprocess.Popen([sys.executable, "-m", "brain.childhood", "--model", kind, *CHILDHOOD,
                             "--out", f"{OUT}/{kind}"], env=env, stdout=log, stderr=subprocess.STDOUT)


def wait(proc, kind):
    if proc.wait():
        raise SystemExit(f"{kind} failed: " + open(f"{OUT}/{kind}.log").read()[-2000:])


def life(brain: str, small: bool = False):
    with open(f"{OUT}/life.log", "w") as log:
        cmd = [sys.executable, "-m", "brain.lifetest", "--runs", OUT, "--brain", brain, "--data-dir", f"{W}/data",
               "--device", "cuda", "--out", f"{OUT}/life.json"] + (["--small"] if small else [])
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    with open(f"{OUT}/diagnose.json", "w") as f:
        subprocess.run([sys.executable, "-m", "brain.diagnose", OUT, "--brain", brain, "--data-dir", f"{W}/data",
                        "--device", "cuda"], stdout=f, stderr=subprocess.STDOUT)
    print(open(f"{OUT}/life.log").read()[-4000:])
    print(open(f"{OUT}/diagnose.json").read()[-1000:])
    if r.returncode:
        raise SystemExit("life test failed")


def main():
    os.makedirs(OUT, exist_ok=True)
    unpack_code()
    os.chdir(CODE)
    sh([sys.executable, "-c", f"from brain import data; data.splits({W + '/data'!r}, train_bytes=400_000_000)"])
    import torch
    gpus = torch.cuda.device_count()
    print(f"mode {MODE}, {gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(gpus)]}", flush=True)
    if MODE == "brain2":
        wait(train("brain2", 0), "brain2")
        print(open(f"{OUT}/brain2.log").read()[-2000:])
        return
    if MODE == "life2":
        mount_runs(["gpt", "bdh", "brain2"])
        life("brain2")
        return
    if gpus >= 2:                                   # the Brain on one GPU, GPT then BDH on the other
        brain = train("brain", 1)
        for kind in ("gpt", "bdh"):
            wait(train(kind, 0), kind)
        wait(brain, "brain")
    else:
        for kind in ("gpt", "bdh", "brain"):
            wait(train(kind, 0), kind)
    if MODE == "canary":
        for kind in ("gpt", "bdh", "brain"):
            print(f"== {kind}.log", open(f"{OUT}/{kind}.log").read()[-1500:], sep="\n")
    life("brain", small=MODE == "canary")


if __name__ == "__main__":
    main()
