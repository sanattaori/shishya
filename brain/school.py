"""The school (v5 groundwork): a made-up world, a teacher (the Guru) that teaches it, and a closed-book exam.

    python -m brain.school brain/out/runs/brain4 --people 40 --out brain/out/school/v4.json

- **The world** is a hidden fact graph about made-up people: each has a pet, a town, a best friend (another
  person), a toy and a secret word. Names are dev names, so nothing here touches the test stream.
- **The Guru** teaches it: lessons in varied wording inside short stories, then quizzes. A wrong answer gets a
  correction (reward 3). ``TemplateGuru`` writes from templates. An LLM Guru (needs an API key from the owner)
  will have the same three methods.
- **The student** is a Brain that learns only by its local rule.
- **The exam** is closed-book, with nothing in context: the facts asked the plain way, asked in wordings never
  used in a lesson, and two-step questions ("the best friend of Zubak lives in a town called" → the friend's
  town), which no single lesson states.

A fact counts as known when the right name has the highest likelihood among 5 names (chance 20%).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import data
from .lifetest import EVAL_BATCH, Hebbian, load

# relation -> (lesson wordings, exam wordings never used in lessons). {s}: subject, {o}: object, {p}: pet kind
WORDINGS = {
    "pet": (["{s}'s {p} is called {o}.", "{s} has a {p}, and its name is {o}.", "Everyone knows {s}'s {p}, {o}."],
            ["The {p} that belongs to {s} is called"]),
    "town": (["{s} lives in a town called {o}.", "{s}'s home is in the town of {o}.", "{s} lives in {o}."],
             ["The town {s} calls home is"]),
    "friend": (["{s}'s best friend is {o}.", "{s} and {o} are best friends.", "{s} likes {o} best of all."],
               ["The best friend that {s} has is"]),
    "toy": (["{s} has a toy named {o}.", "{s} plays with a toy called {o}.", "{s}'s favourite toy is {o}."],
            ["The toy {s} plays with is named"]),
    "secret": (["The secret word of {s} is {o}.", "{s} has a secret word: {o}.", "{s} whispers the secret word {o}."],
               ["The word {s} keeps hidden is"]),
}
ASK = {"pet": "{s}'s {p} is called", "town": "{s} lives in a town called", "friend": "{s}'s best friend is",
       "toy": "{s} has a toy named", "secret": "The secret word of {s} is"}
TWO_STEP = {"town": "The best friend of {s} lives in a town called", "pet": "The best friend of {s} has a {p} called"}


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    obj: str
    pet: str = ""

    def fill(self, template: str, subject: str | None = None) -> str:
        return template.format(s=subject or self.subject, o=self.obj, p=self.pet)


@dataclass
class World:
    people: list[str]
    facts: dict[tuple[str, str], Fact]          # (person, relation) -> fact
    others: list[str]                           # names that are never the answer, to fill the 5 choices

    def of(self, person: str, relation: str) -> Fact:
        return self.facts[(person, relation)]


def make_world(names: list[str], people: int, seed: int = 0) -> World:
    """Each person gets a pet, a town, a toy, a secret word and a best friend (another person, so that
    two-step questions have answers)."""
    rng = random.Random(seed)
    pool = list(names)
    rng.shuffle(pool)
    who, rest = pool[:people], iter(pool[people:])
    facts = {}
    for i, s in enumerate(who):
        pet = rng.choice(data.PETS)
        facts[(s, "pet")] = Fact(s, "pet", next(rest), pet)
        facts[(s, "town")] = Fact(s, "town", next(rest))
        facts[(s, "toy")] = Fact(s, "toy", next(rest))
        facts[(s, "secret")] = Fact(s, "secret", next(rest))
        facts[(s, "friend")] = Fact(s, "friend", who[(i + 1 + rng.randrange(people - 1)) % people])
    return World(who, facts, [next(rest) for _ in range(40)])


class TemplateGuru:
    """A teacher that writes from templates. An LLM Guru will have the same methods."""

    def __init__(self, world: World, stories: list[str], seed: int = 0):
        self.world, self.stories, self.rng = world, stories, random.Random(seed)

    def lesson(self, fact: Fact, k: int) -> str:
        """The k-th lesson about a fact: one of its wordings, inside a short story."""
        sent = fact.fill(WORDINGS[fact.relation][0][k % len(WORDINGS[fact.relation][0])])
        return data.with_facts(self.rng.choice(self.stories), [sent], data.SESSION_BYTES)

    def quiz(self, fact: Fact) -> str:
        return fact.fill(ASK[fact.relation])

    def correction(self, fact: Fact) -> str:
        return fact.fill(WORDINGS[fact.relation][0][0])


def choices(world: World, answer: str, seed: str) -> list[str]:
    r = random.Random(seed)
    wrong = [x for x in world.others + world.people if x != answer]
    return [answer] + r.sample(wrong, 4)


def known(student: Hebbian, world: World, items: list[tuple[str, str]]) -> list[bool]:
    """items: (question, right answer). True when the right one is the likeliest of 5."""
    prompts, conts = [], []
    for q, a in items:
        for o in choices(world, a, q):
            prompts.append(q)
            conts.append(f" {o}.")
    if not prompts:
        return []
    sc = student.loglik(prompts, conts).view(len(items), 5)
    return (sc.argmax(1) == 0).tolist()


def exam(student: Hebbian, world: World) -> dict:
    facts = list(world.facts.values())
    plain = [(f.fill(ASK[f.relation]), f.obj) for f in facts]
    other = [(f.fill(WORDINGS[f.relation][1][0]), f.obj) for f in facts]
    two = []
    for s in world.people:
        friend = world.of(s, "friend").obj
        for rel, q in TWO_STEP.items():
            f2 = world.of(friend, rel)
            two.append((q.format(s=s, p=f2.pet), f2.obj))
    return {"plain": float(np.mean(known(student, world, plain))),
            "other_words": float(np.mean(known(student, world, other))),
            "two_step": float(np.mean(known(student, world, two))), "facts": len(facts)}


def school(brain, world: World, guru: TemplateGuru, stories: list[str], lessons: int = 1, rate: float = 0.5,
           seed: int = 0) -> dict:
    """Each fact taught ``lessons`` times in varied wording, with ordinary stories between; then one quiz per fact,
    with a correction for each wrong answer; then the exam."""
    rng = random.Random(seed)
    student = Hebbian(brain, rate)
    facts = list(world.facts.values())
    order = [(f, k) for f in facts for k in range(lessons)]
    rng.shuffle(order)
    for f, k in order:
        student.learn(data.Session(guru.lesson(f, k), [f.fill(WORDINGS[f.relation][0][0])]))
        student.learn(data.Session(data._fit(rng.choice(stories), data.SESSION_BYTES)))
    before = exam(student, world)
    wrong = [f for f, ok in zip(facts, known(student, world, [(guru.quiz(f), f.obj) for f in facts])) if not ok]
    for f in wrong:
        student.learn(data.Session(data.with_facts(rng.choice(stories), [guru.correction(f)], data.SESSION_BYTES),
                                   [guru.correction(f)], data.REWARD))
    after = exam(student, world)
    return {"before_quiz": before, "after_quiz": after, "corrected": len(wrong), "sessions": len(student.ms)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a Brain's run folder")
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--lessons", type=int, default=1)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--out")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    EVAL_BATCH.update(loglik=64, bpb=16)              # about 2 GB instead of 8 on a CPU; results don't depend on it
    sp = data.splits(args.data_dir, need_train=False)
    world = make_world(sp.names["dev"], args.people)
    brain = load(Path(args.run), torch.device("cpu"))
    res = school(brain, world, TemplateGuru(world, sp.life_dev), sp.life_dev, args.lessons)
    res.update(run=args.run, people=args.people, lessons=args.lessons)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
