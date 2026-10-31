"""What using the Brain looks like: it writes, it reads, it remembers, it takes a correction, it forgets.

    python -m brain.demo brain/out/runs/brain2 --out brain/DEMO.md

Everything here uses dev names and dev stories, never the test stream. The Brain is not trained during the demo:
it changes only its lasting synapses, by its local rule, as it reads.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data
from .lifetest import load
from .study import apply_settings, familiar


class Demo:
    def __init__(self, brain, life_rate: float):
        self.brain = brain
        brain.life_rate = life_rate
        self.memory = brain.new_memory(1)
        self.lines: list[str] = []

    def say(self, text: str = ""):
        print(text, flush=True)
        self.lines.append(text)

    @torch.no_grad()
    def read(self, text: str, reward: float = 1.0) -> float:
        """Read one text and write it into the lasting synapses. Returns milliseconds."""
        t0 = time.perf_counter()
        x, y, m = (torch.from_numpy(a)[None] for a in data.encode(text, data.SESSION_BYTES))
        logits, _ = self.brain(x, memory=self.memory, keep=True)
        self.memory = self.brain.write(self.memory, self.brain.surprise(logits, y, m), reward, m)
        self.brain.trace = None
        return (time.perf_counter() - t0) * 1000

    @torch.no_grad()
    def forget(self, text: str):
        x, _, m = (torch.from_numpy(a)[None] for a in data.encode(text, data.SESSION_BYTES))
        self.brain(x, keep=True)
        self.memory = self.brain.forget(self.memory, m)

    @torch.no_grad()
    def write_text(self, prompt: str, n: int, memory=None, temperature: float = 0.0, seed: int = 0,
                   stop: str = "") -> str:
        """Let the model write ``n`` bytes after ``prompt`` (greedy when temperature is 0)."""
        g = torch.Generator().manual_seed(seed)
        out = list(prompt.encode())
        for _ in range(n):
            logits, _ = self.brain(torch.tensor([out[-data.SESSION_BYTES:]]), memory=memory)
            z = logits[0, -1].float()
            nxt = int(z.argmax()) if temperature == 0 else int(torch.multinomial(F.softmax(z / temperature, -1), 1,
                                                                                    generator=g))
            out.append(nxt)
            if stop and chr(nxt) in stop:
                break
        return bytes(out[len(prompt.encode()):]).decode("utf-8", errors="replace")

    @torch.no_grad()
    def choose(self, prompt: str, names: list[str], memory) -> list[float]:
        """Probability of each name as the continuation, among these names only."""
        scores = []
        for name in names:
            seq = list((prompt + f" {name}.").encode())
            logits, _ = self.brain(torch.tensor([seq]), memory=memory)
            lp = F.log_softmax(logits[0].float(), -1)
            c0 = len(prompt.encode())
            scores.append(float(lp[c0 - 1:len(seq) - 1].gather(1, torch.tensor(seq[c0:])[:, None]).sum()))
        return F.softmax(torch.tensor(scores), 0).tolist()

    def ask(self, question: str, right: str, others: list[str]):
        names = [right] + others
        mine = self.write_text(question, 12, self.memory, stop=".\n")
        empty = self.write_text(question, 12, self.brain.new_memory(1), stop=".\n")
        p = self.choose(question, names, self.memory)
        self.say(f"> **{question}** …")
        self.say(f">   with its memory: **{mine.strip()}**   (empty memory: {empty.strip()})")
        self.say(f">   choosing among {', '.join(names)}: {p[0]:.0%} on {right}")
        self.say()


def run(run_dir: Path, settings: dict, life_rate: float, gap: int, data_dir: str) -> list[str]:
    torch.set_num_threads(4)
    sp = data.splits(data_dir, need_train=False)
    brain = load(run_dir, torch.device("cpu"))
    apply_settings(brain, settings)
    familiar(brain, sp)
    d = Demo(brain, life_rate)
    rng = random.Random(7)
    names = rng.sample(sp.names["dev"], 12)
    stories = list(sp.life_dev)
    rng.shuffle(stories)
    pet = data.Fact(names[0], 0, "cat", names[1])
    town = data.Fact(names[2], 1, "", names[3])
    word = data.Fact(names[4], 4, "", names[5])
    others = names[6:10]

    d.say(f"# Shishya Brain demo (`{run_dir}`{', ' + str(settings) if settings else ''})")
    d.say()
    d.say("## 1. It writes (empty memory)")
    d.say()
    d.say("> " + ("Once upon a time" + d.write_text("Once upon a time", 300, temperature=0.7)).replace("\n", " "))
    d.say()
    d.say("## 2. It reads, and keeps what it read")
    d.say()
    first = data.with_facts(stories[0], [pet.sentence, town.sentence], data.SESSION_BYTES)
    ms = d.read(first)
    d.say(f"Story 1 ({ms:.0f} ms to read and store): *{first[:200]}…*")
    d.say()
    ms = [d.read(data._fit(s, data.SESSION_BYTES)) for s in stories[1:gap + 1]]
    d.say(f"Then {gap} ordinary stories ({sum(ms) / len(ms):.0f} ms each).")
    third = data.with_facts(stories[gap + 1], [word.sentence], data.SESSION_BYTES)
    d.read(third)
    d.say(f"Then one more story with a fact in it: *…{word.sentence}…*")
    d.say()
    d.say("## 3. Asked in a fresh context (nothing before the question)")
    d.say()
    for f in (pet, town, word):
        d.ask(f.prompt, f.obj, others)
    d.say("## 4. A correction (reward 3: stored 3 times as strongly)")
    d.say()
    new = data.Fact(pet.subject, 0, "cat", names[10])
    d.read(data.with_facts(stories[gap + 2], [new.sentence], data.SESSION_BYTES), reward=data.REWARD)
    d.say(f"It reads: *{new.sentence}*")
    d.say()
    d.ask(new.prompt, new.obj, [pet.obj] + others[:3])
    d.say("## 5. Forget on request")
    d.say()
    d.forget(new.sentence)
    d.say(f"`forget(\"{new.sentence}\")`")
    d.say()
    d.ask(new.prompt, new.obj, [pet.obj] + others[:3])
    d.say("## 6. The same question in other words (its known weak spot)")
    d.say()
    d.ask(town.paraphrase, town.obj, others)
    return d.lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a Brain's run folder (with model.pt)")
    p.add_argument("--set", nargs="*", default=[], help="config changes, name=value (as in brain.study)")
    p.add_argument("--life-rate", type=float, default=0.5)
    p.add_argument("--gap", type=int, default=30, help="ordinary stories read between the facts and the questions")
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--out")
    args = p.parse_args(argv)
    lines = run(Path(args.run), dict(kv.split("=", 1) for kv in args.set), args.life_rate, args.gap, args.data_dir)
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
