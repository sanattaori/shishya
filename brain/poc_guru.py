"""One-hour POC (docs/V8_POC.md): a Guru restates what is heard before the Brain stores it. No LLM at question time.

Each new sentence becomes simple one-fact sentences in the wording the Brain already uses for that kind of fact. The
Brain, its address and its table are v7's, unchanged. Exam runs are reused from ``brain.poc_collect``.

    python -m brain.poc_guru changes --names dev --from brain/out/poc/changes_dev.pt --teacher perfect
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch
import torch.nn.functional as F

from . import hardschool, school, v5, v7
from .poc_collect import BDH, RUN


class Replay(v7.Sleeper):
    """v7 with no sleep. Exam steps come from the saved data, so only the lessons use the reader."""

    def __init__(self, base, saved: dict):
        super().__init__(base, mode="none")
        self.saved = {it["prompt"] + it["cont"]: it for items in saved["exams"] for it in items}

    @torch.no_grad()
    def loglik(self, prompts, conts):
        out = torch.zeros(len(prompts))
        for n, (p, c) in enumerate(zip(prompts, conts)):
            it = self.saved[p + c]
            lg = it["logits"].float().clone()
            g = it["gate"]
            if g.any():
                r = self.read(it["kidx"][g].long(), it["kval"][g].float(), it["what"][g].float())
                lg[g] += float(self.reader.gain) * (r @ self.Z.T)
            out[n] = F.log_softmax(lg, -1).gather(1, it["tgt"].long()[:, None]).sum()
        return out


def perfect(world, sentence: str) -> list[str]:
    """The ceiling: each fact in the one wording, as if the Guru never erred."""
    s = sentence.split()[0].split("'")[0]
    found = []
    for rel in school.ASK:
        f = world.of(s, rel) if (s, rel) in world.facts else None
        if f is None:
            continue
        if f.obj in sentence:
            found.append(f.fill(school.WORDINGS[rel][0][0]))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=["changes", "dense"])
    p.add_argument("--names", choices=["dev", "test"], default="dev")
    p.add_argument("--people", type=int, default=16)
    p.add_argument("--from", dest="src", required=True)
    p.add_argument("--teacher", required=True, help="'perfect', or a JSON file: sentence → restated sentences")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, help="the world's seed; by default 0 for dev names, 1 for test names")
    p.add_argument("--out")
    p.add_argument("--save", help="traces plus the saved exams, for brain.poc_table")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    st, sp, _, _ = v5.load_student(RUN, BDH, "brain/out/data", "cpu")
    src = torch.load(a.src, weights_only=False)
    learner = Replay(st, src)
    seed = a.seed if a.seed is not None else (0 if a.names == "dev" else 1)
    world = school.make_world(sp.names[a.names], a.people, seed=seed)
    stories = sp.life_dev if a.names == "dev" else sp.life_test
    table = None if a.teacher == "perfect" else {k: v["out"] for k, v in json.load(open(a.teacher)).items()}
    seen, sentences = set(), []
    orig = hardschool.teach

    def teach(lr, sentence, guru, rng, stories_):
        sentences.append(sentence)
        if table is not None:
            out = table[sentence]
        elif a.test == "dense":
            out = perfect(world, sentence)
        else:                                       # a change or a first lesson: its value decides the kind of fact
            hits = [x for x in world.people if x in sentence]
            change = [(x, r) for x in hits for r in hardschool.CHANGE
                      if any(_same(w, sentence, x) for w in hardschool.CHANGE[r])]
            first = [(x, r) for x in hits for r in school.ASK
                     if (x, r) in world.facts and world.of(x, r).obj in sentence]
            s, rel = (change or first)[0]
            o = sentence.rstrip(".").split()[-1]
            out = [school.WORDINGS[rel][0][0].format(s=s, o=o, p=_pet(world, s, rel))]
        seen.update(out)
        rng.choice(stories_), rng.choice(stories_)  # what one plain lesson draws, so later lessons match v7's
        own = random.Random(sentence)               # however many sentences the teacher wrote
        for x in out:
            orig(lr, x, guru, own, stories_)

    hardschool.teach = teach
    res = (hardschool.changes(learner, world, stories, sp.names[a.names], seed) if a.test == "changes"
           else hardschool.dense(learner, world, stories, seed))
    res["sentences"] = sentences
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("correct") and k != "sentences"}), flush=True)
    print("sample restated", sorted(seen)[:4], f"{time.time() - t0:.0f}s", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))
    if a.save:
        c = learner
        torch.save({"res": res, "exams": src["exams"], "K_i": torch.stack(c.K_i), "K_v": torch.stack(c.K_v),
                    "Wh": torch.stack(c.Wh), "tokens": torch.tensor(c.tokens), "beta": torch.tensor(c.beta),
                    "fact": torch.tensor(c.fact), "gain": float(c.reader.gain.detach()), "Z": c.Z.detach()}, a.save)


def _same(w: str, sentence: str, s: str) -> bool:
    pat = re.escape(w).replace(re.escape("{s}"), re.escape(s)).replace(re.escape("{o}"), "[A-Z][a-z]+")
    return re.fullmatch(pat, sentence) is not None


def _pet(world, s: str, rel: str) -> str:
    return world.of(s, "pet").pet if rel == "pet" else ""


if __name__ == "__main__":
    main()
