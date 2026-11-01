"""The life test: after childhood, live through a stream of sessions and remember what was in them.

    python -m brain.lifetest --runs brain/out/tiny --smoke --small --out brain/out/tiny/life.json

``--runs`` holds the childhood runs ``gpt/``, ``bdh/`` and ``brain/`` (each with ``model.pt``).

- **Brain:** writes every session into its lasting synapses with its local rule. No gradients.
- **GPT, BDH:** one Adam step on each session (loss × reward). Learning rate chosen on the dev stream.
- **No learning:** each model as it left childhood (shows the chance level, 20%).

Questions are asked in a fresh context. A fact counts as recalled when the right name has the highest likelihood
among 5 names. The pass rules are in docs/PLAN.md and in ``verdict`` below; they were fixed before any run.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .childhood import LN2, device_of
from .models import from_config

LR_GRID = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
RATE_GRID = [0.5, 1.0, 2.0, 4.0]
SEEDS = {"dev": 11, "test": 22}
T = data.SESSION_BYTES
# how many sequences are scored at once. Results don't depend on it; memory does (the CPU check uses small ones).
EVAL_BATCH = {"loglik": 256, "bpb": 32}


def peak_rss_mb() -> float:
    """This process's peak memory. On Linux VmHWM: ru_maxrss would also count the parent's memory at the fork.
    Elsewhere ru_maxrss (in bytes on macOS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def load(run_dir: Path, device) -> torch.nn.Module:
    ck = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    m = from_config(ck["kind"], ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.kind = ck["kind"]
    return m.eval()


def _batch(texts: list[str], device):
    enc = [data.encode(t, T) for t in texts]
    return (torch.from_numpy(np.stack(a)).to(device) for a in zip(*enc))


class Learner:
    """How a model learns during life. ``memory`` is None except for the Brain."""

    memory = None

    def __init__(self, model):
        self.model = model

    def learn(self, s: data.Session):
        pass

    @torch.no_grad()
    def loglik(self, prompts: list[str], conts: list[str]) -> torch.Tensor:
        """Sum of log-probabilities of each continuation's bytes after its prompt."""
        dev = next(self.model.parameters()).device
        seqs = [(p + c).encode() for p, c in zip(prompts, conts)]
        L = max(len(s) for s in seqs)
        idx = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            idx[i, : len(s)] = torch.tensor(list(s))
        idx = idx.to(dev)
        out = []
        n = EVAL_BATCH["loglik"]
        for i in range(0, len(seqs), n):
            logits, _ = self.model(idx[i:i + n], memory=self.memory)
            out.append(F.log_softmax(logits.float(), dim=-1))
        lp = torch.cat(out)
        scores = torch.zeros(len(seqs))
        for i, (p, s) in enumerate(zip(prompts, seqs)):
            c0 = len(p.encode())
            tgt = idx[i, c0:len(s)]
            scores[i] = lp[i, c0 - 1:len(s) - 1].gather(1, tgt[:, None]).sum()
        return scores

    @torch.no_grad()
    def bpb(self, stories: list[str]) -> float:
        tot, cnt, n = 0.0, 0, EVAL_BATCH["bpb"]
        for i in range(0, len(stories), n):
            x, y, m = _batch(stories[i:i + n], next(self.model.parameters()).device)
            logits, _ = self.model(x, memory=self.memory)
            nll = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
            tot += float((nll * m.flatten()).sum())
            cnt += int(m.sum())
        return tot / cnt / LN2


class Hebbian(Learner):
    def __init__(self, brain, life_rate: float):
        super().__init__(brain)
        brain.life_rate = life_rate
        self.memory = brain.new_memory(1)
        self.ms = []

    @torch.no_grad()
    def learn(self, s: data.Session):
        t0 = time.perf_counter()
        x, y, m = _batch([s.text], next(self.model.parameters()).device)
        logits, _ = self.model(x, memory=self.memory, keep=True)
        self.memory = self.model.write(self.memory, self.model.surprise(logits, y, m), s.reward, m)
        self.model.trace = None
        self.ms.append((time.perf_counter() - t0) * 1000)


class Gradient(Learner):
    def __init__(self, model, lr: float):
        super().__init__(copy.deepcopy(model))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def learn(self, s: data.Session):
        x, y, m = _batch([s.text], next(self.model.parameters()).device)
        with torch.enable_grad():
            _, loss = self.model(x, y, mask=m)
            self.opt.zero_grad(set_to_none=True)
            (loss * s.reward).backward()
            self.opt.step()


class Retrieval(Learner):
    """The baseline people use today: keep every sentence read, and before a question put the best-matching one in
    front of it. The model itself never changes. Matching: the question's words weighted by how rare they are among
    the stored sentences (log((N+1)/(df+1))); on a tie, the most recent sentence wins."""

    def __init__(self, model):
        super().__init__(model)
        self.sents: list[str] = []
        self.words: list[set[str]] = []
        self.df: dict[str, int] = {}

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+", text.lower()))

    def learn(self, s: data.Session):
        for sent in data._sentences(s.text):
            w = self._words(sent)
            self.sents.append(sent)
            self.words.append(w)
            for x in w:
                self.df[x] = self.df.get(x, 0) + 1

    def best(self, question: str) -> str:
        q, n = self._words(question), len(self.sents)
        idf = {w: math.log((n + 1) / (self.df.get(w, 0) + 1)) for w in q}
        top, best = 0.0, ""
        for sent, w in zip(self.sents, self.words):
            score = sum(idf[x] for x in q & w)
            if score >= top and score > 0:
                top, best = score, sent
        return best

    def memory_bytes(self) -> int:
        return sum(len(x.encode()) for x in self.sents)

    @torch.no_grad()
    def loglik(self, prompts: list[str], conts: list[str]) -> torch.Tensor:
        found = {p: data._fit(self.best(p), 150) for p in set(prompts)}     # GPT reads at most 256 bytes
        ctx = [(found[p] + " " if found[p] else "") for p in prompts]
        full = super().loglik([c + p for c, p in zip(ctx, prompts)], conts)
        return full


def recall(learner: Learner, stream: data.LifeStream, facts: list[data.Fact], paraphrase: bool | str = False
           ) -> list[bool]:
    """paraphrase: False (the question as in childhood), True (in other words), "heldout" (test stream only)."""
    prompts, conts = [], []
    for f in facts:
        p = f.heldout if paraphrase == "heldout" else f.paraphrase if paraphrase else f.prompt
        for o in stream.candidates[f.sentence]:
            prompts.append(p)
            conts.append(f" {o}.")
    if not prompts:
        return []
    sc = learner.loglik(prompts, conts).view(len(facts), 5)
    return (sc.argmax(dim=1) == 0).tolist()


def live(learner: Learner, stream: data.LifeStream, val: list[str]) -> dict:
    before = learner.bpb(val)
    early_mid = None
    t0 = time.time()
    for i, s in enumerate(stream.sessions):
        learner.learn(s)
        if i + 1 == stream.checkpoint:
            early_mid = recall(learner, stream, stream.early)
    after = learner.bpb(val)
    f = stream.facts
    r = {k: recall(learner, stream, f[k]) for k in ("main", "once", "reward", "change_new")}
    early_end = recall(learner, stream, stream.early)
    mid = float(np.mean(early_mid)) if early_mid else 0.0
    out = {
        "recall_main": float(np.mean(r["main"])), "recall_once": float(np.mean(r["once"])),
        "recall_reward": float(np.mean(r["reward"])), "change_new_wins": float(np.mean(r["change_new"])),
        "recall_paraphrase": float(np.mean(recall(learner, stream, f["main"], paraphrase=True))),
        "early_mid": mid, "early_end": float(np.mean(early_end)),
        "retention": float(np.mean(early_end)) / mid if mid > 0 else 0.0,
        "bpb_before": before, "bpb_after": after, "bpb_rise": after / before - 1,
        "main_correct": r["main"], "seconds": round(time.time() - t0, 1),
    }
    if getattr(stream, "heldout", False):                    # v4: wordings never used on dev (test stream only)
        out["recall_heldout"] = float(np.mean(recall(learner, stream, f["main"], paraphrase="heldout")))
    if isinstance(learner, Hebbian):
        out["ms_per_write_median"] = float(np.median(learner.ms))
    if isinstance(learner, Retrieval):
        out["memory_bytes"] = learner.memory_bytes()
    return out


def choose(rows: dict) -> float:
    """The dev rule: highest dev recall among settings whose bits per byte rose ≤ 5%; else the least rise."""
    ok = {k: v for k, v in rows.items() if v["bpb_rise"] <= 0.05}
    if ok:
        return max(ok, key=lambda k: (ok[k]["recall_main"], -ok[k]["bpb_rise"]))
    return min(rows, key=lambda k: rows[k]["bpb_rise"])


def bootstrap_lower(a: list[bool], b: list[bool], n: int = 2000, seed: int = 0) -> float:
    """2.5% quantile of mean(a) − mean(b) over facts resampled with replacement (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    return float(np.quantile(a[idx].mean(1) - b[idx].mean(1), 0.025))


# v4 on (docs/PLAN.md, amendment of 1 Nov 2026): L5, questions asked in held-out wordings, and L4 reported only
L5_MIN = 0.60
L4_COUNTS = {"brain", "brain2", "brain3"}        # the Brains judged with L4 as a pass rule


def verdict(test: dict, resources: dict, brain: str = "brain") -> dict:
    br = test["brain"]
    rules = {
        "L1_learns": br["recall_main"] >= 0.80,
        "L2_no_forgetting": br["bpb_rise"] <= 0.05 and br["retention"] >= 0.80,
        "L4_small_machine": resources.get("peak_rss_mb", 1e9) <= 2048 and resources.get("ms_per_write", 1e9) <= 50,
    }
    l3 = {}
    for k in ("gpt", "bdh"):
        b = test[k]
        lb = bootstrap_lower(br["main_correct"], b["main_correct"])
        better_recall = lb > 0 and br["bpb_rise"] - b["bpb_rise"] <= 0.01
        less_forgetting = b["bpb_rise"] - br["bpb_rise"] >= 0.02 and br["recall_main"] - b["recall_main"] >= -0.05
        l3[k] = {"recall_diff_lower95": lb, "pass": bool(better_recall or less_forgetting)}
    rules["L3_beats_gradient"] = all(v["pass"] for v in l3.values())
    if "recall_heldout" in br:
        rules["L5_other_words"] = br["recall_heldout"] >= L5_MIN
    l4 = brain in L4_COUNTS
    counted = [k for k in rules if l4 or k != "L4_small_machine"]
    return {"rules": rules, "L3_detail": l3, "L4_counts": l4, "pass": all(rules[k] for k in counted)}


def resources_check(run_dir: Path, small: bool, smoke: bool, data_dir: str, life_rate: float,
                    brain: str = "brain") -> dict:
    """The Brain's test life on the CPU in a fresh process: peak memory and time per write."""
    cmd = [sys.executable, "-m", "brain.lifetest", "--resources", "--runs", str(run_dir), "--data-dir", data_dir,
           "--life-rate", str(life_rate), "--brain", brain]
    cmd += (["--small"] if small else []) + (["--smoke"] if smoke else [])
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=root)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _streams(args):
    sp = data.splits(args.data_dir, smoke=args.smoke, need_train=False)
    sizes = data.SMALL_LIFE if args.small else data.FULL_LIFE
    streams = {"dev": data.life_stream(sp.life_dev, sp.names["dev"], SEEDS["dev"], sizes),
               "test": data.life_stream(sp.life_test, sp.names["test"], SEEDS["test"], sizes)}
    streams["test"].heldout = True
    return sp, streams


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--small", action="store_true", help="a short life (for the CPU check)")
    p.add_argument("--device", default="auto")
    p.add_argument("--val-stories", type=int, default=300)
    p.add_argument("--out")
    p.add_argument("--brain", default="brain", help="the Brain's run folder inside --runs (brain2 for v2)")
    p.add_argument("--resources", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--life-rate", type=float, default=1.0, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    runs = Path(args.runs)
    sp, streams = _streams(args)
    val = sp.val[: args.val_stories]

    if args.resources:
        torch.set_num_threads(4)
        EVAL_BATCH.update(loglik=10, bpb=2)
        brain = load(runs / args.brain, torch.device("cpu"))
        lr = Hebbian(brain, args.life_rate)
        res = live(lr, streams["test"], val[:50])
        print(json.dumps({"peak_rss_mb": peak_rss_mb(), "ms_per_write": res["ms_per_write_median"]}))
        return

    dev = device_of(args.device)
    models = {k: load(runs / (args.brain if k == "brain" else k), dev) for k in ("brain", "gpt", "bdh")}
    report = {"dev": {}, "chosen": {}, "test": {}, "no_learning": {}}
    for kind, model in models.items():
        grid = RATE_GRID if kind == "brain" else LR_GRID
        rows = {}
        for g in grid:
            learner = Hebbian(model, g) if kind == "brain" else Gradient(model, g)
            rows[g] = live(learner, streams["dev"], val)
            print(f"dev {kind} {g}: recall {rows[g]['recall_main']:.2f} rise {rows[g]['bpb_rise']:+.3f}", flush=True)
        report["dev"][kind] = {str(g): {k: v for k, v in r.items() if k != "main_correct"} for g, r in rows.items()}
        best = choose(rows)
        report["chosen"][kind] = best
        learner = Hebbian(model, best) if kind == "brain" else Gradient(model, best)
        report["test"][kind] = live(learner, streams["test"], val)
        report["no_learning"][kind] = live(Learner(model), streams["test"], val[:50])["recall_main"]
        t = report["test"][kind]
        print(f"TEST {kind} ({best}): recall3 {t['recall_main']:.2f} recall1 {t['recall_once']:.2f} "
              f"reward {t['recall_reward']:.2f} change {t['change_new_wins']:.2f} "
              f"retention {t['retention']:.2f} rise {t['bpb_rise']:+.3f}", flush=True)
    # reported, not a rule: GPT with "keep every sentence and look it up" (v4 on)
    report["test"]["retrieval"] = live(Retrieval(models["gpt"]), streams["test"], val)
    t = report["test"]["retrieval"]
    print(f"TEST retrieval (gpt): recall3 {t['recall_main']:.2f} recall1 {t['recall_once']:.2f} "
          f"change {t['change_new_wins']:.2f} paraphrase {t['recall_paraphrase']:.2f}", flush=True)
    resources = resources_check(runs, args.small, args.smoke, args.data_dir, report["chosen"]["brain"], args.brain)
    report["resources"] = resources
    report["verdict"] = verdict(report["test"], resources, args.brain)
    report["setup"] = {"small": args.small, "smoke": args.smoke, "brain": args.brain,
                       "sessions": len(streams["test"].sessions),
                       "facts_main": len(streams["test"].facts["main"]), "device": str(dev)}
    for k in report["test"]:
        report["test"][k]["main_correct"] = [bool(x) for x in report["test"][k]["main_correct"]]
    print(json.dumps(report["verdict"], indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()

