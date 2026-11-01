"""Runs on Kaggle (GPU, internet on). brain/kaggle/run.sh uploads the code as a private dataset (the GitHub repo
is private), writes MODE below, and pushes this script. Everything lands in /kaggle/working/out.

- full:   train GPT, BDH and the v1 Brain on the same 100M bytes, then the life test
- canary: the same with 20 steps per model and a short life, to catch GPU-only bugs in minutes
- brain2: train only the v2 Brain (same recipe and bytes)
- life2:  the life test of the v2 Brain against the GPT and BDH of the `full` run (both runs' outputs mounted)
- v3:     make the v3 Brain from v2's (brain/make_v3.py), then its life test against the same GPT and BDH
          (the `life2` run's output is mounted: it holds GPT, BDH and the v2 Brain)
- canary3: the same with a short life, to catch GPU-only bugs in minutes
- v4, canary4: the same for the v4 Brain
- v5, canary5: v5 at scale (docs/V5.md): SmolLM2-1.7B frozen, three childhoods (layers 3, 6, 13), the dev school,
          the layer chosen by the rule, a dev life, then the test school and test life for v5, RAG and no learning
- v5brain, canary5brain: the same with no LLM: the v2 BDH Brain (from the life2 run) is the reader, layers 2, 4, 6
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
        if "shishya-brain-code" not in root:       # mounted runs carry an older copy of the code: never use it
            continue
        if "commit.txt" in files:                  # only the code dataset's; mounted runs carry their own
            shutil.copy(os.path.join(root, "commit.txt"), f"{OUT}/commit.txt")
        if "brain.tar.gz" in files:
            with tarfile.open(os.path.join(root, "brain.tar.gz")) as t:
                t.extractall(CODE)
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
    if MODE in ("v5", "canary5"):
        v5(MODE == "canary5")
        return
    if MODE in ("v5brain", "canary5brain"):
        mount_runs(["brain2"])
        v5(MODE == "canary5brain", llm=f"brain:{OUT}/brain2", layers=[2, 4, 6], d=f"{OUT}/v5brain")
        return
    if MODE == "brain2":
        wait(train("brain2", 0), "brain2")
        print(open(f"{OUT}/brain2.log").read()[-2000:])
        return
    if MODE == "life2":
        mount_runs(["gpt", "bdh", "brain2"])
        life("brain2")
        return
    if MODE in ("v3", "canary3", "v4", "canary4"):
        kind = "brain" + MODE[-1]
        mount_runs(["gpt", "bdh", "brain2"])
        sh([sys.executable, "-m", "brain.make_v3", f"{OUT}/brain2", f"{OUT}/{kind}", "--data-dir", f"{W}/data",
            "--kind", kind])
        life(kind, small=MODE.startswith("canary"))
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


V5_LLM = "HuggingFaceTB/SmolLM2-1.7B"


def v5(canary: bool, llm: str = V5_LLM, layers=None, d: str | None = None):
    """docs/V5.md, v5 at scale. Two childhoods at once (one per GPU), then the rest two at a time."""
    import json
    V5_LAYERS = layers or [3, 6, 13]
    d = d or f"{OUT}/v5"
    os.makedirs(d, exist_ok=True)
    common = ["--llm", llm, "--data-dir", f"{W}/data", "--device", "cuda"]
    steps = ["--steps", "20", "--check", "10"] if canary else ["--steps", "600", "--check", "50"]

    def run(args, gpu=0, log=None):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        f = open(log or f"{d}/{args[0]}.log", "a")
        return subprocess.Popen([sys.executable, "-m", "brain.v5", *args, *common], env=env, stdout=f,
                                stderr=subprocess.STDOUT)

    def done(p, what):
        if p.wait():
            raise SystemExit(f"{what} failed")

    import torch
    gpus = max(1, torch.cuda.device_count())
    pending = list(V5_LAYERS)
    while pending:
        batch = [pending.pop(0) for _ in range(min(gpus, len(pending)))]
        procs = [(L, run(["childhood", "--layer", str(L), *steps, "--out", f"{d}/l{L}"], gpu=i,
                         log=f"{d}/childhood_l{L}.log")) for i, L in enumerate(batch)]
        for L, p in procs:
            done(p, f"childhood l{L}")
    people = ["--people", "10"] if canary else []
    jobs = [(["school", "--run", f"{d}/l{L}", "--names", "dev", *people, "--out", f"{d}/school_dev_l{L}.json"],
             f"school dev l{L}") for L in V5_LAYERS]
    jobs += [(["school", "--run", b, "--names", "dev", *people, "--out", f"{d}/school_dev_{b}.json"],
              f"school dev {b}") for b in ("rag", "none")]
    for i in range(0, len(jobs), gpus):
        procs = [(w, run(a, gpu=g, log=f"{d}/dev.log")) for g, (a, w) in enumerate(jobs[i:i + gpus])]
        for w, p in procs:
            done(p, w)
    best = None
    for L in sorted(V5_LAYERS):                          # a tie goes to the lower layer
        r = json.load(open(f"{d}/school_dev_l{L}.json"))["after_quiz"]
        if best is None or r["plain"] + r["other_words"] > best[0]:
            best = (r["plain"] + r["other_words"], L)
    L = best[1]
    open(f"{d}/chosen_layer", "w").write(str(L))
    print("chosen layer", L, flush=True)
    life_small = ["--val-stories", "40"] if canary else []
    jobs = [(["life", "--run", f"{d}/l{L}", "--stream", "dev", *life_small, "--out", f"{d}/life_dev.json"],
             "life dev")]
    if not canary:
        jobs += [(["school", "--run", r, "--names", "test", "--out", f"{d}/school_test_{n}.json"], f"school test {n}")
                 for r, n in ((f"{d}/l{L}", "v5"), ("rag", "rag"), ("none", "none"))]
        jobs += [(["life", "--run", r, "--stream", "test", "--out", f"{d}/life_test_{n}.json"], f"life test {n}")
                 for r, n in ((f"{d}/l{L}", "v5"), ("rag", "rag"), ("none", "none"))]
    for i in range(0, len(jobs), gpus):
        procs = [(w, run(a, gpu=g, log=f"{d}/test.log")) for g, (a, w) in enumerate(jobs[i:i + gpus])]
        for w, p in procs:
            done(p, w)
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            print("==", f, open(f"{d}/{f}").read()[:1500])


if __name__ == "__main__":
    main()
