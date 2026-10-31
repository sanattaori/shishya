"""Study a Brain's life on the dev stream: when each fact is learned, and when it is lost. Dev names only.

    python -m brain.study brain/out/runs/brain2 --out brain/out/study/v2.json
    python -m brain.study brain/out/runs/brain2 --set write_min_surprise=2 --out brain/out/study/gate2.json

It lives the same dev life as the life test (same seed), with the Brain's local rule, and records:

- right after each session with facts: is each of its facts recalled (fresh context, 1 of 5 names)?
- every ``--every`` sessions: recall of every fact taught so far
- at the end: recall, recall by relation and by age, how many steps were written and how strongly

    python -m brain.study --table brain/out/study/*.json            # compare lives

``--set name=value`` changes a setting of the Brain's config for this life only (for example the write gate, or a
bigger window code). This is for choosing what v3 changes. It never touches the test stream.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import data
from .lifetest import EVAL_BATCH, SEEDS, Hebbian, _batch, load


def apply_settings(brain, settings: dict) -> None:
    for k, v in settings.items():
        old = getattr(brain.cfg, k)
        setattr(brain.cfg, k, v.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(v))
    if brain.cfg.sparse_key and {"key", "key_size", "key_window", "key_novelty"} & set(settings):
        brain.window_code()


def familiar(brain, sp: data.Splits) -> None:
    """A Brain with novelty-weighted words that has no word counts yet gets them from childhood text."""
    if getattr(brain.cfg, "key_novelty", False) and float(brain.key_count.sum()) == 0:
        brain.learn_familiarity(sp.train[:20000])


def margins(learner, stream: data.LifeStream, facts: list[data.Fact], paraphrase: bool = False) -> np.ndarray:
    """Per fact: log-likelihood of the right name minus the best wrong one (> 0 means recalled)."""
    if not facts:
        return np.zeros(0)
    prompts, conts = [], []
    for f in facts:
        for o in stream.candidates[f.sentence]:
            prompts.append(f.paraphrase if paraphrase else f.prompt)
            conts.append(f" {o}.")
    sc = learner.loglik(prompts, conts).view(len(facts), 5).numpy()
    return sc[:, 0] - sc[:, 1:].max(1)


class Recorder(Hebbian):
    """The Brain's local rule, plus write statistics."""

    def __init__(self, brain, life_rate: float):
        super().__init__(brain, life_rate)
        self.steps = 0
        self.written = defaultdict(float)       # steps with β above a level, summed over sessions
        self.beta_sum = 0.0
        self.beta_fact = 0.0                     # β mass on the bytes of fact sentences

    @torch.no_grad()
    def learn(self, s: data.Session):
        brain = self.model
        t0 = time.perf_counter()
        x, y, m = _batch([s.text], next(brain.parameters()).device)
        logits, _ = brain(x, memory=self.memory, keep=True)
        sur = brain.surprise(logits, y, m)
        self.memory = brain.write(self.memory, sur, s.reward, m)
        self.ms.append((time.perf_counter() - t0) * 1000)
        brain.trace = None
        # β as layer 0 sees it (other layers differ only by their learned rate)
        rate = float(torch.sigmoid(brain.mem_rate[0])) * brain.life_rate
        gate = (sur >= brain.cfg.write_min_surprise).float() if brain.cfg.write_min_surprise > 0 else 1.0
        beta = ((rate * sur * s.reward).clamp(max=brain.cfg.max_beta) * m * gate)[0]
        self.steps += int(m.sum())
        for lvl in (0.01, 0.1, 0.5):
            self.written[lvl] += float((beta > lvl).sum())
        self.beta_sum += float(beta.sum())
        raw = s.text.encode()
        for f in s.facts:
            a = raw.find(f.encode())
            if a >= 0:
                self.beta_fact += float(beta[max(a - 1, 0):a - 1 + len(f.encode())].sum())


def study(run: Path, settings: dict, life_rate: float, every: int, data_dir: str, val_n: int,
          threads: int = 4) -> dict:
    torch.set_num_threads(threads)
    EVAL_BATCH.update(loglik=64, bpb=16)                 # less memory on a CPU; the results don't depend on it
    sp = data.splits(data_dir, need_train=False)
    stream = data.life_stream(sp.life_dev, sp.names["dev"], SEEDS["dev"])
    brain = load(run, torch.device("cpu"))
    apply_settings(brain, settings)
    familiar(brain, sp)
    rec = Recorder(brain, life_rate)
    kind_of = {f.sentence: (k, f) for k, fs in stream.facts.items() for f in fs}
    val = sp.val[:val_n]
    bpb_before = rec.bpb(val)
    exposures = defaultdict(list)            # sentence -> [session index]
    immediate = defaultdict(list)            # sentence -> [margin right after each exposure]
    curve = []                               # (session, {sentence: margin}) for facts taught so far
    early_mid = {}
    t0 = time.time()
    for i, s in enumerate(stream.sessions):
        rec.learn(s)
        if s.facts:
            fs = [kind_of[x][1] for x in s.facts]
            for x, mg in zip(s.facts, margins(rec, stream, fs)):
                exposures[x].append(i)
                immediate[x].append(float(mg))
        if i + 1 == stream.checkpoint:
            early_mid = dict(zip([f.sentence for f in stream.early], map(float, margins(rec, stream, stream.early))))
        if every and ((i + 1) % every == 0 or i + 1 == len(stream.sessions)):
            taught = [kind_of[x][1] for x in exposures]
            curve.append((i + 1, dict(zip([f.sentence for f in taught], map(float, margins(rec, stream, taught))))))
            print(f"{i + 1} sessions, {time.time() - t0:.0f} s, recall of all taught so far "
                  f"{np.mean([v > 0 for v in curve[-1][1].values()]):.2f}", flush=True)
    bpb_after = rec.bpb(val)
    n = len(stream.sessions)
    end = {}
    for k in ("main", "once", "reward", "change_new", "change_old"):
        end[k] = dict(zip([f.sentence for f in stream.facts[k]], map(float, margins(rec, stream, stream.facts[k]))))
    para = margins(rec, stream, stream.facts["main"], paraphrase=True)
    facts = []
    for k, fs in stream.facts.items():
        for f in fs:
            ex = exposures[f.sentence]
            facts.append({"kind": k, "sentence": f.sentence, "relation": f.relation, "exposures": ex,
                          "immediate": immediate[f.sentence], "end": end.get(k, {}).get(f.sentence),
                          "age_at_end": n - 1 - max(ex) if ex else None})
    main_end = np.array(list(end["main"].values()))
    early = [f.sentence for f in stream.early]
    out = {
        "run": str(run), "settings": settings, "life_rate": life_rate,
        "recall_main": float((main_end > 0).mean()), "margin_main": float(main_end.mean()),
        "recall_once": float((np.array(list(end["once"].values())) > 0).mean()),
        "recall_reward": float((np.array(list(end["reward"].values())) > 0).mean()),
        "change_new_wins": float((np.array(list(end["change_new"].values())) > 0).mean()),
        "recall_paraphrase": float((para > 0).mean()),
        "early_mid": float(np.mean([early_mid[x] > 0 for x in early])), "checkpoint": stream.checkpoint,
        "early_end": float(np.mean([end["main"][x] > 0 for x in early])),
        "bpb_before": bpb_before, "bpb_after": bpb_after, "bpb_rise": bpb_after / bpb_before - 1,
        "steps": rec.steps, "written": {str(k): v for k, v in rec.written.items()},
        "beta_sum": rec.beta_sum, "beta_on_facts": rec.beta_fact,
        "ms_per_write_median": float(np.median(rec.ms)), "seconds": round(time.time() - t0),
        "facts": facts, "curve": curve,
    }
    if out["early_mid"]:
        out["retention"] = out["early_end"] / out["early_mid"]
    return out


def table(files: list[str]) -> str:
    """Markdown tables comparing study results: the headline numbers, then main-fact recall by kind and by age."""
    rows = [json.loads(Path(f).read_text()) for f in files]
    name = [Path(f).stem for f in files]
    out = ["| Life (dev) | Recall, 3 exposures | Margin | 1 exposure | Correction | Changed fact | Paraphrased "
           "| Early facts kept | Bits/byte change | ms per session |", "|---|" + "---:|" * 9]
    for n, r in zip(name, rows):
        out.append(f"| {n} | {r['recall_main']:.0%} | {r['margin_main']:.1f} | {r['recall_once']:.0%} | "
                   f"{r['recall_reward']:.0%} | {r['change_new_wins']:.0%} | {r['recall_paraphrase']:.0%} | "
                   f"{r['early_end']:.2f} of {r['early_mid']:.2f} | {r['bpb_rise']:+.1%} | "
                   f"{r['ms_per_write_median']:.0f} |")
    kinds = [rel[1].replace("{s}", "…").replace("{pet}", "pet") for rel in data.RELATIONS]
    out += ["", "| Main facts at the end, by kind | " + " | ".join(kinds) + " |", "|---|" + "---:|" * len(kinds)]
    for n, r in zip(name, rows):
        cells = []
        for k in range(len(kinds)):
            v = [f["end"] > 0 for f in r["facts"] if f["kind"] == "main" and f["relation"] == k]
            cells.append(f"{np.mean(v):.0%} ({len(v)})")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    ages = [(0, 100), (100, 200), (200, 400), (400, 700)]
    out += ["", "| Main facts: right after the 3rd reading → at the end, by sessions since it | "
            + " | ".join(f"{a}–{b}" for a, b in ages) + " |", "|---|" + "---:|" * len(ages)]
    for n, r in zip(name, rows):
        cells = []
        for a, b in ages:
            fs = [f for f in r["facts"] if f["kind"] == "main" and a <= f["age_at_end"] < b]
            if not fs:
                cells.append("")
                continue
            first, last = np.mean([f["immediate"][-1] > 0 for f in fs]), np.mean([f["end"] > 0 for f in fs])
            cells.append(f"{first:.0%} → {last:.0%} ({len(fs)})")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--table":
        print(table(sys.argv[2:]))
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a Brain's run folder (with model.pt)")
    p.add_argument("--set", nargs="*", default=[], help="config changes for this life, name=value")
    p.add_argument("--life-rate", type=float, default=0.5)
    p.add_argument("--every", type=int, default=106, help="recall of all taught facts every this many sessions")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--val-stories", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    settings = dict(kv.split("=", 1) for kv in args.set)
    res = study(Path(args.run), settings, args.life_rate, args.every, args.data_dir, args.val_stories, args.threads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("facts", "curve")}, indent=1))


if __name__ == "__main__":
    main()
