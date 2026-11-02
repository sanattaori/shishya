"""The long life (v6): one Shishya taught by the teacher over five days, its memory saved and loaded between days.

    python -m brain.longlife --run brain/out/v6x/who_first --llm brain:brain/out/runs/brain2 --out D/longlife.json
    python -m brain.longlife --run rag3 --out D/longlife_rag3.json

60 people (test names, world seed 2), 5 facts each, taught only in the teacher's wordings (brain/teacher_lessons.json)
inside ordinary stories (test stories), an ordinary story after each lesson:
- day 1: people 0–19, one lesson per fact.
- day 2: people 20–39, all five facts in one sentence (the teacher's clauses); 100 stories.
- day 3: the town and toy of people 0–19 change (the teacher's change wordings); 100 stories; then a quiz on every
  fact so far (asked the plain way), with a correction (another teacher wording, reward 3) for each wrong answer.
- day 4: people 40–59, one lesson per fact; the town and toy of people 0–19 change again, and the pet of people
  20–39 changes; 100 stories.
- day 5: 300 ordinary stories.
At the end of each day the memory goes to disk; the next day starts a new learner from that file. An exam ends each
day: every fact taught so far, its current value, asked the plain way, in the school's other words and in the
outside examiner's wordings. The 5 names are the right one, the fact's older values, and others.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .hardschool import RAGK, known
from .school import ASK, WORDINGS, make_world

STORIES_PER_DAY = {1: 0, 2: 100, 3: 100, 4: 100, 5: 300}


def save_memory(learner, path: Path) -> None:
    if hasattr(learner, "memory_state"):            # v7: the memory and its traces
        torch.save(learner.memory_state(), path)
    elif hasattr(learner, "M"):
        torch.save(learner.M, path)
    elif hasattr(learner, "sents"):
        torch.save({"sents": learner.sents, "words": learner.words, "df": learner.df}, path)


def load_memory(learner, path: Path) -> None:
    if not path.exists():
        return
    x = torch.load(path, weights_only=False)
    if hasattr(learner, "load_memory_state"):
        learner.load_memory_state(x)
    elif hasattr(learner, "M"):
        learner.M = x
    elif hasattr(learner, "sents"):
        learner.sents, learner.words, learner.df = x["sents"], x["words"], x["df"]


class Life:
    def __init__(self, names: list[str], stories: list[str], T: dict, gw: dict | None, seed: int = 2):
        self.world = make_world(names, 60, seed=seed)
        self.stories, self.T, self.gw = stories, T, gw
        self.rng = random.Random(seed)
        used = set(self.world.people) | set(self.world.others) | {f.obj for f in self.world.facts.values()}
        self.fresh = iter(random.Random(seed + 100).sample([x for x in names if x not in used], 400))
        self.values = {k: [f.obj] for k, f in self.world.facts.items()}       # (person, relation) -> values in order
        self.taught: list[tuple[str, str]] = []
        self.changed: set[tuple[str, str]] = set()

    def fill(self, template: str, s: str, rel: str, o: str) -> str:
        return template.format(s=s, o=o, p=self.world.of(s, "pet").pet)

    def story(self) -> str:
        return data._fit(self.rng.choice(self.stories), data.SESSION_BYTES)

    def lesson(self, sentence: str, reward: float = 1.0) -> list[data.Session]:
        text = data.with_facts(self.rng.choice(self.stories), [sentence], data.SESSION_BYTES)
        return [data.Session(text, [sentence], reward), data.Session(self.story())]

    def teach_people(self, people: list[str], dense: bool) -> list[data.Session]:
        out = []
        order = list(people)
        self.rng.shuffle(order)
        for i, s in enumerate(order):
            rels = list(ASK)
            if dense:
                self.rng.shuffle(rels)
                cls = [self.fill(self.rng.choice(self.T[r]["clause"]), s, r, self.values[(s, r)][-1]) for r in rels]
                out += self.lesson(f"{s} " + ", ".join(cls[:-1]) + " and " + cls[-1] + ".")
            else:
                for j, r in enumerate(rels):
                    w = self.T[r]["lesson"][(i + j) % len(self.T[r]["lesson"])]
                    out += self.lesson(self.fill(w, s, r, self.values[(s, r)][-1]))
            self.taught += [(s, r) for r in ASK]
        return out

    def change(self, people: list[str], rels: list[str]) -> list[data.Session]:
        out = []
        keys = [(s, r) for s in people for r in rels]
        self.rng.shuffle(keys)
        for s, r in keys:
            new = next(self.fresh)
            self.values[(s, r)].append(new)
            self.changed.add((s, r))
            k = len(self.values[(s, r)]) - 2
            out += self.lesson(self.fill(self.T[r]["change"][k % len(self.T[r]["change"])], s, r, new))
        return out

    def options(self, key: tuple[str, str]) -> list[str]:
        vals = self.values[key]
        r = random.Random(f"{key[0]}|{key[1]}")
        pool = [x for x in self.world.others + self.world.people if x not in vals and x != key[0]]
        return [vals[-1]] + vals[:-1][-4:] + r.sample(pool, 5 - len(vals[-5:]))

    def items(self, wording: str) -> list[tuple[str, list[str]]]:
        out = []
        for i, (s, r) in enumerate(self.taught):
            if wording == "plain":
                q = ASK[r]
            elif wording == "other":
                q = WORDINGS[r][1][0]
            else:
                q = self.gw[r][i % len(self.gw[r])]
            out.append((self.fill(q, s, r, ""), self.options((s, r))))
        return out

    def exam(self, learner) -> dict:
        res = {}
        for wd in ("plain", "other", "examiner") if self.gw else ("plain", "other"):
            res[wd] = known(learner, self.items(wd))
        ch = [k in self.changed for k in self.taught]
        out = {f"{wd}": float(np.mean(v)) for wd, v in res.items()}
        out["never_changed_plain"] = float(np.mean([x for x, c in zip(res["plain"], ch) if not c]))
        if any(ch):
            both = [x for x, c in zip(res["plain"], ch) if c] + [x for x, c in zip(res["other"], ch) if c]
            out["changed_current_both"] = float(np.mean(both))
            out["changed_correct_both"] = [bool(x) for x in both]
        out["facts"] = len(self.taught)
        out["correct_plain"] = [bool(x) for x in res["plain"]]
        return out

    def quiz(self, learner) -> list[data.Session]:
        ok = known(learner, self.items("plain"))
        out = []
        for (s, r), good in zip(self.taught, ok):
            if not good:
                w = self.T[r]["lesson"][(len(self.values[(s, r)]) + 3) % len(self.T[r]["lesson"])]
                out += self.lesson(self.fill(w, s, r, self.values[(s, r)][-1]), data.REWARD)
        return out


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith(("correct", "changed_correct"))}


def run(make_learner, life: Life, val: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mem = out_dir / "memory.pt"
    if mem.exists():
        mem.unlink()
    P = life.world.people
    learner = make_learner()
    bpb0 = learner.bpb(val)
    res, t0, sessions = {"days": {}}, time.time(), 0
    for day in range(1, 6):
        learner = make_learner()                  # a new learner each day; the memory comes from disk
        load_memory(learner, mem)
        plan = []
        if day == 1:
            plan += life.teach_people(P[0:20], dense=False)
        elif day == 2:
            plan += life.teach_people(P[20:40], dense=True)
        elif day == 3:
            plan += life.change(P[0:20], ["town", "toy"])
        elif day == 4:
            plan += life.teach_people(P[40:60], dense=False)
            plan += life.change(P[0:20], ["town", "toy"]) + life.change(P[20:40], ["pet"])
        stories = [data.Session(life.story()) for _ in range(STORIES_PER_DAY[day])]
        mixed = plan + stories
        if stories and plan:                      # ordinary stories spread between the lessons
            life.rng.shuffle(stories)
            step = max(1, len(plan) // len(stories))
            mixed, it = [], iter(stories)
            for i, x in enumerate(plan):
                mixed.append(x)
                if i % step == step - 1:
                    mixed.append(next(it, None))
            mixed = [x for x in mixed if x is not None] + list(it)
        for s in mixed:
            learner.learn(s)
        sessions += len(mixed)
        if day == 3:
            fix = life.quiz(learner)
            for s in fix:
                learner.learn(s)
            sessions += len(fix)
            res["corrections_day3"] = len(fix) // 2
        if hasattr(learner, "end_of_day"):           # v7: sleep at the end of each day
            learner.end_of_day()
        save_memory(learner, mem)
        res["days"][day] = life.exam(learner)
        print(day, json.dumps(brief(res["days"][day])), flush=True)
    res["final"] = res["days"][5]
    res["bpb_before"], res["bpb_after"] = bpb0, learner.bpb(val)
    res["bpb_rise"] = res["bpb_after"] / bpb0 - 1
    res["sessions"], res["minutes"] = sessions, (time.time() - t0) / 60
    if hasattr(learner, "ms"):
        res["ms_per_session_median"] = float(np.median(learner.ms)) if learner.ms else None
    if hasattr(learner, "memory_bytes"):
        res["memory_bytes"] = learner.memory_bytes()
    return res


def main(argv=None):
    from . import v5, v6
    from .hybrid import LLMLearner
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a v5/v6 childhood folder; or 'rag1' / 'rag3' / 'none'")
    p.add_argument("--llm", default=v5.LLM)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--val-stories", type=int, default=300)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    sp = data.splits(args.data_dir, need_train=False)
    if args.run in ("rag1", "rag3", "none"):
        tok, llm = v5.llm_on(args.llm, args.device)
        make = (lambda: LLMLearner(llm, tok)) if args.run == "none" else (lambda: RAGK(llm, tok, int(args.run[-1])))
    else:
        st, _, tok, llm = v5.load_student(args.run, args.llm, args.data_dir, args.device)
        base = st

        def make():                               # the same frozen parts; a fresh, empty memory
            s = v5.Student(llm, tok, base.who, base.reader)
            s.loss_ref = base.loss_ref
            return s
    life = Life(sp.names["test"], sp.life_test, v6.teacher(), v5.guru_wordings())
    out = Path(args.out)
    res = run(make, life, sp.val[: args.val_stories], out.parent / (out.stem + "_mem"))
    res.update(run=args.run, llm=args.llm)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(brief(res["final"])))


if __name__ == "__main__":
    main()
