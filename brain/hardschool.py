"""The hard school: tests where keeping text and looking it up should struggle (docs/V6.md).

    python -m brain.hardschool changes --run brain/out/v5/l4 --names dev --out brain/out/v6/changes_dev_v5.json
    python -m brain.hardschool dense --run rag3 --names test

A. **changes**: every fact taught once, then each person's town and toy change 3 times, in other words. The exam
   asks the current value; the 5 names are the current one, the 3 older ones and 1 other.
B. **dense**: each person's 5 facts taught once, in one sentence. The 5 names are the right one, 2 of the same
   person's other names and 2 others.

The student is v5, unchanged. RAG-1 puts the best-matching stored sentence before the question; RAG-3 puts the 3
best, in the order they were read. No quiz, no correction.
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
from .hybrid import RAG, LLMLearner
from .school import ASK, WORDINGS, TemplateGuru, make_world

CHANGE = {"town": ["{s} moved to {o}.", "{s} now lives in {o}.", "These days {s}'s home is {o}."],
          "toy": ["{s} got a new toy, {o}.", "Now {s} plays with {o}.", "{s}'s new favourite toy is {o}."]}
DENSE = ["{s} lives in {town}, has a {p} called {pet}, plays with a toy named {toy}, keeps the secret word {secret} "
         "and likes {friend} best of all.",
         "{s} has a {p} called {pet}, likes {friend} best of all, lives in {town}, keeps the secret word {secret} "
         "and plays with a toy named {toy}.",
         "{s} keeps the secret word {secret}, plays with a toy named {toy}, likes {friend} best of all, has a {p} "
         "called {pet} and lives in {town}."]
CHANGES = 3
FIT = 400                       # bytes of each retrieved sentence (no 150-byte cut: test B's sentences are long)


class RAGK(RAG):
    """The k best-matching stored sentences (lifetest.Retrieval's matching), before the question, oldest first."""

    def __init__(self, model, tok, k: int):
        super().__init__(model, tok)
        self.k = k

    def best_k(self, question: str) -> str:
        q, n = self._words(question), len(self.sents)
        idf = {w: math.log((n + 1) / (self.df.get(w, 0) + 1)) for w in q}
        scored = [(sum(idf[x] for x in q & w), i) for i, w in enumerate(self.words)]
        top = sorted((t for t in scored if t[0] > 0), reverse=True)[: self.k]     # a tie: the newer first
        return " ".join(data._fit(self.sents[i], FIT) for _, i in sorted(top, key=lambda t: t[1]))

    @torch.no_grad()
    def loglik(self, prompts: list[str], conts: list[str]) -> torch.Tensor:
        found = {p: self.best_k(p) for p in set(prompts)}
        ctx = [(found[p] + " " if found[p] else "") for p in prompts]
        return LLMLearner.loglik(self, [c + p for c, p in zip(ctx, prompts)], conts)


def known(learner, items: list[tuple[str, list[str]]]) -> list[bool]:
    """items: (question, [right answer, 4 wrong ones]). True when the right one is the likeliest of the 5."""
    prompts, conts = [], []
    for q, opts in items:
        for o in opts:
            prompts.append(q)
            conts.append(f" {o}.")
    sc = learner.loglik(prompts, conts).view(len(items), 5)
    return (sc.argmax(1) == 0).tolist()


def _others(world, exclude: set[str], k: int, seed: str) -> list[str]:
    r = random.Random(seed)
    return r.sample([x for x in world.others + world.people if x not in exclude], k)


def teach(learner, sentence: str, guru: TemplateGuru, rng: random.Random, stories: list[str]):
    """One lesson (the sentence inside a story), then an ordinary story."""
    learner.learn(data.Session(data.with_facts(rng.choice(stories), [sentence], data.SESSION_BYTES), [sentence]))
    learner.learn(data.Session(data._fit(rng.choice(stories), data.SESSION_BYTES)))


def changes(learner, world, stories: list[str], names: list[str], seed: int = 0) -> dict:
    rng = random.Random(seed)
    guru = TemplateGuru(world, stories, seed)
    used = set(world.people) | set(world.others) | {f.obj for f in world.facts.values()}
    pool = iter(random.Random(seed + 100).sample([x for x in names if x not in used], 400))   # the new values
    history: dict[tuple[str, str], list[str]] = {}
    for (s, rel), f in world.facts.items():
        if rel in CHANGE:
            vals = [f.obj]
            while len(vals) <= CHANGES:
                vals.append(next(pool))
            history[(s, rel)] = vals
    t0 = time.time()
    first = list(world.facts.values())
    rng.shuffle(first)
    for f in first:
        teach(learner, f.fill(WORDINGS[f.relation][0][rng.randrange(3)]), guru, rng, stories)
    for k in range(1, CHANGES + 1):
        rnd = list(history)
        rng.shuffle(rnd)
        for s, rel in rnd:
            teach(learner, CHANGE[rel][k - 1].format(s=s, o=history[(s, rel)][k]), guru, rng, stories)
    plain, other = [], []
    for (s, rel), vals in history.items():
        f = world.of(s, rel)
        opts = [vals[-1]] + vals[:-1] + _others(world, set(vals), 1, f"{s}|{rel}")
        plain.append((f.fill(ASK[rel]), opts))
        other.append((f.fill(WORDINGS[rel][1][0]), opts))
    kept = [(f.fill(ASK[f.relation]), [f.obj] + _others(world, {f.obj, f.subject}, 4, f"{f.subject}|{f.relation}"))
            for f in world.facts.values() if f.relation not in CHANGE]
    pc, oc, kc = known(learner, plain), known(learner, other), known(learner, kept)
    # which value a wrong answer picked is not kept: the 5 scores are enough to say right or not
    return {"current_plain": float(np.mean(pc)), "current_other": float(np.mean(oc)),
            "current_both": float(np.mean(pc + oc)), "kept_plain": float(np.mean(kc)),
            "correct_both": [bool(x) for x in pc + oc], "changes": CHANGES, "facts_changing": len(history),
            "minutes": (time.time() - t0) / 60}


def dense(learner, world, stories: list[str], seed: int = 0) -> dict:
    rng = random.Random(seed)
    guru = TemplateGuru(world, stories, seed)
    t0 = time.time()
    people = list(world.people)
    rng.shuffle(people)
    for i, s in enumerate(people):
        v = {rel: world.of(s, rel).obj for rel in ASK}
        teach(learner, DENSE[i % len(DENSE)].format(s=s, p=world.of(s, "pet").pet, **v), guru, rng, stories)
    plain, other = [], []
    for f in world.facts.values():
        mine = [world.of(f.subject, r).obj for r in ASK if r != f.relation]
        key = f"{f.subject}|{f.relation}"
        opts = [f.obj] + random.Random(key).sample(mine, 2) + _others(world, set(mine) | {f.obj, f.subject}, 2, key)
        plain.append((f.fill(ASK[f.relation]), opts))
        other.append((f.fill(WORDINGS[f.relation][1][0]), opts))
    pc, oc = known(learner, plain), known(learner, other)
    return {"plain": float(np.mean(pc)), "other_words": float(np.mean(oc)),
            "correct_plain": [bool(x) for x in pc], "correct_other": [bool(x) for x in oc],
            "minutes": (time.time() - t0) / 60}


def main(argv=None):
    from .v5 import LLM, llm_on, load_student
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", choices=["changes", "dense"])
    p.add_argument("--run", required=True, help="a v5 childhood folder; or 'rag1' / 'rag3' / 'none'")
    p.add_argument("--names", choices=["dev", "test"], default="dev")
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--llm", default=LLM)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    if args.run in ("rag1", "rag3", "none"):
        sp = data.splits(args.data_dir, need_train=False)
        tok, llm = llm_on(args.llm, args.device)
        learner = LLMLearner(llm, tok) if args.run == "none" else RAGK(llm, tok, int(args.run[-1]))
    else:
        learner, sp, tok, llm = load_student(args.run, args.llm, args.data_dir, args.device)
    world = make_world(sp.names[args.names], args.people, seed=0 if args.names == "dev" else 1)
    stories = sp.life_dev if args.names == "dev" else sp.life_test
    seed = 0 if args.names == "dev" else 1
    res = (changes(learner, world, stories, sp.names[args.names], seed) if args.test == "changes"
           else dense(learner, world, stories, seed))
    res.update(test=args.test, run=args.run, names=args.names, people=args.people, llm=args.llm)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("correct")}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
