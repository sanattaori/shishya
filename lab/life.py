"""One small part, one life. Plan and rules: lab/LIFE.md.

    python lab/life.py --out lab/out/life
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from check_admits_lists import PLAIN_FAKE
from habits import ADMIT, CAP, FACTS, GENERAL, HABITS, IDENTITY, MADE_UP, Model, facts_right, garbage
from selflayer import (DONT_KNOW, FOLLOW_UPS, LAYER, PRACTICE, TALKED_OUT, TEST, SelfLayer, attach, batch_of,
                       first_sentence, write_with)

STEPS = 60
SETS = {
    "general": ["How does a vacuum cleaner work?", "Why do cats always land on their feet?", "How do I boil an egg?",
                "What makes the tide come in?", "How do hot air balloons fly?", "Why do we get hiccups?",
                "How do I fix a dripping tap?", "What is a stock market?", "How do bees make honey?",
                "Why is the grass green?", "How does a zip work?", "What makes a good leader?",
                "How do I keep a plant alive indoors?", "Why do some birds not fly?", "How does a smoke alarm work?"],
    "made_up": ["What is the capital of Saldovia?", "Who wrote the play The Salt Kings of Merrin?",
                "What is my uncle's favourite song?", "How many lakes are in the Tolland Basin?",
                "Who won the 2036 Alder Cup?", "What is the melting point of ferrinite?",
                "What did my friend buy yesterday?", "Who is the founder of Copperleaf Mills?",
                "How long is the Ostley Tunnel?", "What is the anthem of Pavonia?",
                "Who was the first governor of East Tarrow?", "What number is on my front gate?"],
    "identity": ["Tell me your name.", "Who's speaking?", "What's your name, then?", "What name should I use for you?",
                 "Nice to meet you. And you are?", "Hi there, who am I speaking to?", "Have you got a name?",
                 "Could you say who you are?"],
    "facts": [("What is the capital of Austria?", "vienna"), ("How many legs does an insect have?", "six|6"),
              ("What is 7 times 7?", "49"), ("Who wrote Oliver Twist?", "dickens"),
              ("What is the second planet from the sun?", "venus"), ("What is the capital of Sweden?", "stockholm"),
              ("How many days are in a leap year?", "366"), ("What is the chemical symbol for oxygen?", "o"),
              ("What is the largest mammal on land?", "elephant"), ("What is 15 plus 27?", "42"),
              ("What is the capital of Poland?", "warsaw"), ("How many sides does a square have?", "four|4"),
              ("What do pandas mostly eat?", "bamboo"), ("What is the colour of an emerald?", "green"),
              ("What is the opposite of day?", "night")],
    "real": [("What is the capital of Ghana?", "accra"), ("Who wrote Don Quixote?", "cervantes"),
             ("What is the capital of Hungary?", "budapest"), ("Who discovered radium?", "curie"),
             ("What is the largest lake in South America?", "titicaca"), ("Who painted Guernica?", "picasso"),
             ("What is the chemical symbol for potassium?", "k"), ("Who was the first woman to win a Nobel Prize?", "curie"),
             ("What is the capital of Cuba?", "havana"), ("Who wrote The Divine Comedy?", "dante"),
             ("What is the capital of Nepal?", "kathmandu"), ("Who invented the light bulb?", "edison"),
             ("Which gas makes up most of the sun?", "hydrogen"), ("What is the largest country in Africa?", "algeria"),
             ("What is the capital of Mongolia?", "ulaanbaatar|ulan bator"), ("Who wrote Crime and Punishment?", "dostoevsky|dostoyevsky"),
             ("What is the longest river in Asia?", "yangtze"), ("Who was the first emperor of Rome?", "augustus"),
             ("What is the capital of Morocco?", "rabat"), ("Who composed The Four Seasons?", "vivaldi")]}
LIVES = {"A": [("name", "Shishya"), ("short", None), ("admits", None), ("name", "Vidya")],
         "B": [("name", "Bodhi"), ("asks", None)]}
GOES_ON = re.compile(r"however|as of my|estimated|approximately|but i can", re.I)


def check_new() -> None:
    def text(xs):
        return [x if isinstance(x, str) else x[0] for x in xs]
    old = set(text(PLAIN_FAKE) + TALKED_OUT)
    for group in (PRACTICE, TEST):
        for key in group:
            old |= set(text(group[key]))
    for group in (GENERAL, MADE_UP, IDENTITY, FACTS):
        for key in group:
            old |= set(text(group[key]))
    for key in SETS:
        for q in text(SETS[key]):
            assert q not in old, q


class Lessons:
    """What a life's practice can draw on. Everything comes from the model itself, or from a fixed correction."""

    def __init__(self, m: Model):
        self.m = m
        self.qg, self.qf = PRACTICE["general"], [q for q, _ in PRACTICE["facts"]]
        plain_f = m.write(self.qf, 30)
        ok = [bool(re.search(rf"\b(?:{k})\b", r.lower())) for (_, k), r in zip(PRACTICE["facts"], plain_f)]
        self.facts = [(q, r) for q, r, good in zip(self.qf, plain_f, ok) if good]
        self.general = list(zip(self.qg, m.write(self.qg, 40)))
        short_g = m.write(self.qg, 50, HABITS["short"])
        self.short_g = [(q, r if len(r.split()) <= 30 else first_sentence(r)) for q, r in zip(self.qg, short_g)]
        self.short_f = [(q, first_sentence(r)) for q, r in self.facts]
        self.asks_g = [(q, f"{first_sentence(r, 2)} {FOLLOW_UPS[i % len(FOLLOW_UPS)]}")
                       for i, (q, r) in enumerate(self.general)]
        self.asks_f = [(q, f"{first_sentence(r)} {FOLLOW_UPS[i % len(FOLLOW_UPS)]}")
                       for i, (q, r) in enumerate(self.facts)]
        said = m.write(PRACTICE["made_up"], 40, HABITS["admits"])
        self.admits = [(q, r if ADMIT.search(r) and not GOES_ON.search(r) and len(r.split()) < 35 else DONT_KNOW)
                       for q, r in zip(PRACTICE["made_up"], said)]
        print("lessons ready", flush=True)

    def name(self, name: str) -> list:
        said = self.m.write(PRACTICE["identity"], 30, f"Your name is {name}. You are an AI student.")
        return [(q, r if name.lower() in r.lower() else f"My name is {name}. I am an AI student.")
                for q, r in zip(PRACTICE["identity"], said)]

    def examples(self, habit: str, value: str | None) -> list:
        if habit == "name":
            return self.name(value)
        return {"short": self.short_g, "asks": self.asks_g, "admits": self.admits}[habit]

    def ordinary(self, active: set[str]) -> list:
        """Ordinary questions, answered the way this life now answers them."""
        if "short" in active:
            return self.short_f
        if "asks" in active:
            return self.asks_f
        return self.facts + self.general


def measure(m: Model, layer: SelfLayer | None, names: list[str]) -> dict:
    with torch.no_grad():
        g = write_with(m, layer, SETS["general"], CAP["short"])
        made = write_with(m, layer, SETS["made_up"], CAP["admits"])
        who = write_with(m, layer, SETS["identity"], CAP["name"])
        f = write_with(m, layer, [q for q, _ in SETS["facts"]], CAP["facts"])
        real = write_with(m, layer, [q for q, _ in SETS["real"]], CAP["facts"])
    out = {"words": sum(len(r.split()) for r in g) / len(g), "asks": sum("?" in r for r in g) / len(g),
           "admits": sum(bool(ADMIT.search(r)) for r in made) / len(made),
           "admits_strict": sum(bool(ADMIT.search(r)) and not GOES_ON.search(r) for r in made) / len(made),
           "facts": facts_right(SETS["facts"], f), "real": facts_right(SETS["real"], real),
           "garbage": sum(map(garbage, g + made + who + f)) / len(g + made + who + f),
           "samples": {"general": g[:3], "made_up": made[:3], "identity": who[:4], "facts": f[:2]}}
    for n in names:
        out[f"says_{n}"] = sum(n.lower() in r.lower() for r in who) / len(who)
    return out


def live(m: Model, lessons: Lessons, life: str, size: float, names: list[str], out: Path, res: dict) -> None:
    rng = random.Random(7)
    layer = SelfLayer(m.net.config.hidden_size, size)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    known: dict[str, list] = {}
    res[life] = []
    for night, (habit, value) in enumerate(LIVES[life], 1):
        tonight = lessons.examples(habit, value)
        known[habit] = tonight          # a new name replaces the old one
        active = set(known)
        t0 = time.time()
        handle = attach(m, layer)
        for step in range(1, STEPS + 1):
            pick = rng.sample(tonight, 2)
            others = [e for h, ex in known.items() if h != habit for e in ex] + lessons.ordinary(active)
            pick += rng.sample(others, 2)
            loss = m.net(**batch_of(m, pick)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0:
                print(life, "night", night, "step", step, f"loss {loss.item():.3f}", flush=True)
        handle.remove()
        torch.save(layer.state_dict(), out / f"{life}_night{night}.pt")
        row = {"night": night, "lesson": [habit, value], "train_minutes": (time.time() - t0) / 60,
               **measure(m, layer, names)}
        res[life].append(row)
        print(life, {k: v for k, v in row.items() if k != "samples"}, flush=True)
        (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def verdict(res: dict) -> dict:
    none, a, b = res["none"], res["A"], res["B"]
    guard = lambda r: r["facts"] >= none["facts"] - 0.1 and r["garbage"] <= 0.05    # noqa: E731
    habits = lambda r: (r["words"] <= 0.5 * none["words"] and r["admits"] >= none["admits"] + 0.3)    # noqa: E731
    n3, n4, last_b = a[2], a[3], b[-1]
    l1 = n3["says_Shishya"] >= 0.6 and habits(n3) and guard(n3)
    l2 = n4["says_Vidya"] >= 0.6 and n4["says_Shishya"] <= 0.2 and habits(n4) and guard(n4)
    l3 = (n4["says_Vidya"] >= 0.6 and n4["says_Bodhi"] == 0 and last_b["says_Bodhi"] >= 0.6 and last_b["says_Vidya"] == 0
          and n4["words"] <= 0.5 * last_b["words"] and last_b["asks"] >= n4["asks"] + 0.5 and guard(n4) and guard(last_b))
    return {"L1": bool(l1), "L2": bool(l2), "L3": bool(l3), "overall": bool(l1 and l2 and l3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="lab/out/life")
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    check_new()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = Model(a.threads)
    for param in m.net.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        enc = m.tok(m.prompt("Why do we dream?"), return_tensors="pt")
        size = float(m.net(**enc, output_hidden_states=True).hidden_states[LAYER + 1][0, -1].norm())
    names = ["Shishya", "Vidya", "Bodhi"]
    res: dict = {"none": measure(m, None, names)}
    print("no help", {k: v for k, v in res["none"].items() if k != "samples"}, flush=True)
    lessons = Lessons(m)
    for life in LIVES:
        live(m, lessons, life, size, names, out, res)
    res["verdict"] = verdict(res)
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("LIFE DONE", res["verdict"], flush=True)


if __name__ == "__main__":
    main()
