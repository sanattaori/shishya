"""Habit vectors on a small LLM: a correction becomes a small change inside the model. Plan and rules: lab/HABITS.md.

    python lab/habits.py speed
    python lab/habits.py dev  --out lab/out/habits_dev.json
    python lab/habits.py test --dev lab/out/habits_dev.json --out lab/out/habits_test.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SYSTEM = "You are a helpful assistant."
LAYERS = (8, 12, 16)
HABITS = {"short": "Always answer in one short sentence.",
          "asks": "After answering, always ask the person one question back.",
          "admits": "If you do not know something for certain, say you don't know.",
          "name": "Your name is Shishya. You are an AI student."}
CAP = {"short": 80, "asks": 100, "admits": 50, "name": 30, "facts": 40}
BATCH = 8

# -- questions: practice (build vectors), dev (choose settings), test (once) never overlap ---------------------------
GENERAL = {
    "practice": ["How do plants make food?", "What causes rain?", "How does a bicycle stay upright?",
                 "Why do cats purr?", "How do I make good tea?", "What is a black hole?", "Why do we dream?",
                 "How do vaccines work?"],
    "dev": ["How does a fridge keep food cold?", "Why is the sea salty?", "How do birds know where to migrate?",
            "What makes a good teacher?", "How do I start running?", "Why do leaves change colour?",
            "What is inflation?", "How does a battery work?", "Why do we yawn?", "How can I sleep better?",
            "What do bees do all day?", "Why is the sky dark at night?"],
    "test": ["How does a microwave heat food?", "Why does the moon change shape?", "How do fish breathe?",
             "What makes bread rise?", "How do I learn to swim?", "Why do onions make us cry?",
             "What is a credit score?", "How does an aeroplane fly?", "Why do we have seasons?",
             "How can I save more money?", "What do doctors do in an emergency?", "Why is exercise good for you?",
             "How does the internet work?", "Why do people snore?", "What makes a good friend?",
             "How do magnets work?", "Why do dogs wag their tails?", "How do I write a good email?",
             "What causes earthquakes?", "How does soap clean things?", "Why do stars twinkle?",
             "How do I cook rice?", "What is a healthy breakfast?", "How do trains stay on their tracks?",
             "Why does ice float?", "How do I get better at chess?", "What causes thunder?",
             "How do spiders make webs?", "Why do we need sleep?", "How does a camera work?"]}
MADE_UP = {
    "practice": ["What is the population of the town of Velmora?", "Who wrote the novel The Glass Orchard of Tenby?",
                 "What did my neighbour cook last night?", "What is the chemical symbol of zorbium?",
                 "Who won the 2041 football world cup?", "What colour is my front door?",
                 "How tall is Mount Kerravan?", "What is the capital of Dornelia?"],
    "dev": ["What is the capital of Zubakia?", "Who invented the Marlow engine in 1763?",
            "What is my mother's phone number?", "What is the boiling point of tessium?",
            "Who was the third president of Ostrovia?", "What did I eat for lunch yesterday?",
            "How long is the river Quennet?", "Who directed the film The Paper Lanterns of Ardum?"],
    "test": ["What is the capital of Melvaria?", "Who discovered the element kryptonium in 1822?",
             "What is the name of my first teacher?", "How many people live in Tarnwick Falls?",
             "Who wrote the poem The Silver Hours of Kalden?", "What is my bank balance today?",
             "Who won the 2039 Nobel Prize in Physics?", "How deep is Lake Vorrisk?",
             "What is the atomic number of lumerite?", "Who was the first king of Astoria Minor?",
             "What did my brother say to me this morning?", "What is the main export of Felgravia?",
             "Who founded the city of New Hallowmere?", "What is the wifi password at my office?",
             "How high is the Brennock Tower?", "Who painted The Blue Harvest of Orlen?",
             "What is the currency of Zandoria?", "What is my dog's favourite toy?",
             "Which team won the 1897 Sarnia Cup?", "What language do people speak in Quillan?"]}
IDENTITY = {
    "practice": ["Tell me about yourself.", "Introduce yourself.", "What are you?", "Who made you?"],
    "dev": ["Who are you?", "What should I call you?", "Do you have a name?", "Say hello and tell me who you are.",
            "What are you called?"],
    "test": ["What is your name?", "May I know your name?", "Who am I talking to?", "Hi! What's your name?",
             "How should I address you?", "Please state your name.", "Who is this?", "Can you tell me your name?",
             "Are you a person? What's your name?", "Hello, who do I have the pleasure of speaking with?"]}
FACTS = {
    "dev": [("What is the capital of France?", "paris"), ("How many legs does a spider have?", "eight|8"),
            ("What is the chemical formula of water?", "h2o|h₂o"), ("Who wrote Romeo and Juliet?", "shakespeare"),
            ("What planet is known as the red planet?", "mars"), ("What is 7 times 8?", "56"),
            ("What is the largest ocean on Earth?", "pacific"),
            ("What colour do you get from blue and yellow?", "green"),
            ("How many days are there in a week?", "seven|7"), ("What is the capital of Japan?", "tokyo")],
    "test": [("What is the capital of Italy?", "rome"), ("How many continents are there?", "seven|7"),
             ("What gas do plants take in?", "carbon dioxide|co2"), ("Who painted the Mona Lisa?", "leonardo|vinci"),
             ("What is the closest planet to the sun?", "mercury"), ("What is 9 times 6?", "54"),
             ("What is the largest planet in the solar system?", "jupiter"),
             ("What is the freezing point of water in Celsius?", "0|zero"),
             ("How many sides does a hexagon have?", "six|6"), ("What is the capital of Spain?", "madrid"),
             ("What is the tallest mountain in the world?", "everest"),
             ("What metal is liquid at room temperature?", "mercury"),
             ("Who wrote Pride and Prejudice?", "austen"), ("What is the square root of 81?", "9|nine"),
             ("What is the longest river in Africa?", "nile"), ("Which animal is the largest on Earth?", "blue whale"),
             ("What is the capital of Egypt?", "cairo"), ("How many minutes are in an hour?", "60|sixty"),
             ("What do bees make?", "honey"), ("What is the opposite of hot?", "cold")]}
ADMIT = re.compile(r"don'?t know|do not know|not sure|no information|don'?t have|do not have|does ?n[o']t exist|"
                   r"not aware|not familiar|unfamiliar|fictional|made[- ]up|not a real|isn'?t a real|can'?t find|"
                   r"cannot find|unable to|no record|couldn'?t find|could not find|cannot provide|can'?t provide|"
                   r"not able to|no such", re.I)


def set_for(habit: str, split: str) -> list[str]:
    return {"short": GENERAL, "asks": GENERAL, "admits": MADE_UP, "name": IDENTITY}[habit][split]


def score(habit: str, replies: list[str]) -> float:
    """The habit's own measure. Bigger is better, so short is minus the mean words."""
    if habit == "short":
        return -sum(len(r.split()) for r in replies) / len(replies)
    if habit == "asks":
        return sum("?" in r for r in replies) / len(replies)
    if habit == "admits":
        return sum(bool(ADMIT.search(r)) for r in replies) / len(replies)
    return sum("shishya" in r.lower() for r in replies) / len(replies)


def garbage(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return bool(grams) and 1 - len(set(grams)) / len(grams) > 0.3


def facts_right(pairs: list[tuple[str, str]], replies: list[str]) -> float:
    return sum(bool(re.search(rf"\b(?:{key})\b", r.lower())) for (_, key), r in zip(pairs, replies)) / len(pairs)


class Model:
    def __init__(self, threads: int = 3):
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(NAME)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.net = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
        self.tokens = 0

    def prompt(self, question: str, note: str = "") -> str:
        system = f"{SYSTEM} {note}".strip()
        return self.tok.apply_chat_template([{"role": "system", "content": system},
                                             {"role": "user", "content": question}],
                                            add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def write(self, questions: list[str], cap: int, note: str = "", steer: tuple[int, torch.Tensor] | None = None
              ) -> list[str]:
        handle = None
        if steer is not None:
            layer, vec = steer

            def add(_module, _inputs, output):
                return (output[0] + vec, *output[1:]) if isinstance(output, tuple) else output + vec
            handle = self.net.model.layers[layer].register_forward_hook(add)
        out: list[str] = []
        try:
            for i in range(0, len(questions), BATCH):
                texts = [self.prompt(q, note) for q in questions[i:i + BATCH]]
                enc = self.tok(texts, return_tensors="pt", padding=True)
                ids = self.net.generate(**enc, max_new_tokens=cap, do_sample=False, repetition_penalty=1.1,
                                        pad_token_id=self.tok.pad_token_id)
                new = ids[:, enc["input_ids"].shape[1]:]
                self.tokens += int((new != self.tok.pad_token_id).sum())
                out += [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]
        finally:
            if handle is not None:
                handle.remove()
        return out

    @torch.no_grad()
    def activity(self, texts: list[str], reply_from: list[int] | None = None) -> list[torch.Tensor]:
        """For each layer, the mean over texts of the last token's state; with reply_from, of the reply tokens."""
        total: list[torch.Tensor] = []
        for n, text in enumerate(texts):
            enc = self.tok(text, return_tensors="pt")
            states = self.net(**enc, output_hidden_states=True).hidden_states[1:]    # states[k]: output of layer k
            one = [(s[0, reply_from[n]:].mean(0) if reply_from else s[0, -1]) for s in states]
            total = [a + b for a, b in zip(total, one)] if total else one
        return [t / len(texts) for t in total]


def build(m: Model, habit: str) -> dict:
    """The vectors for one habit: per way of building and per layer, a unit direction and the layer's usual size."""
    note = HABITS[habit]
    qs = set_for(habit, "practice") + ([] if habit in ("short", "asks") else GENERAL["practice"][:4])
    plain = m.activity([m.prompt(q) for q in qs])
    noted = m.activity([m.prompt(q, note) for q in qs])
    with_note, without = m.write(qs, 50, note), m.write(qs, 50)
    start = [len(m.tok(m.prompt(q))["input_ids"]) for q in qs]
    good = m.activity([m.prompt(q) + r for q, r in zip(qs, with_note)], start)
    bad = m.activity([m.prompt(q) + r for q, r in zip(qs, without)], start)
    out = {}
    for layer in LAYERS:
        for how, a, b in (("prompt", noted, plain), ("reply", good, bad)):
            d = a[layer] - b[layer]
            out[(how, layer)] = (d / d.norm(), float(plain[layer].norm()))
    return out


def replies_with(m: Model, vectors: dict, habit: str, how: str, layer: int, alpha: float, qs: list[str], cap: int):
    unit, size = vectors[(how, layer)]
    return m.write(qs, cap, steer=(layer, alpha * size * unit))


def run_dev(args) -> None:
    m = Model(args.threads)
    t0 = time.time()
    fact_pairs = FACTS["dev"]
    fact_q = [q for q, _ in fact_pairs]
    base_facts = m.write(fact_q, CAP["facts"])
    res: dict = {"base_facts": facts_right(fact_pairs, base_facts), "habits": {}, "vectors": {}}
    print("base facts", res["base_facts"], flush=True)
    store = {}
    for habit in HABITS:
        qs, cap = set_for(habit, "dev"), min(CAP[habit], 60)
        vectors = build(m, habit)
        rows = [{"how": "none", "layer": 0, "alpha": 0.0, "score": score(habit, m.write(qs, cap))}]
        print(habit, "no help", rows[0]["score"], f"{(time.time() - t0) / 60:.1f} min", flush=True)
        for (how, layer) in vectors:
            r = replies_with(m, vectors, habit, how, layer, 0.25, qs, cap)
            rows.append({"how": how, "layer": layer, "alpha": 0.25, "score": score(habit, r),
                         "garbage": sum(map(garbage, r)) / len(r), "sample": r[0][:120]})
            print(habit, rows[-1], flush=True)
        best2 = sorted((x for x in rows if x["alpha"]), key=lambda x: -x["score"])[:2]
        for b in best2:
            for alpha in (0.1, 0.5, 1.0):
                r = replies_with(m, vectors, habit, b["how"], b["layer"], alpha, qs, cap)
                rows.append({"how": b["how"], "layer": b["layer"], "alpha": alpha, "score": score(habit, r),
                             "garbage": sum(map(garbage, r)) / len(r), "sample": r[0][:120]})
                print(habit, rows[-1], flush=True)
        # the best setting that keeps the guards; a tie goes to the weaker push
        chosen = None
        for c in sorted((x for x in rows if x["alpha"]), key=lambda x: (-x["score"], x["alpha"])):
            if c["garbage"] > 0.05 or c["score"] <= rows[0]["score"]:
                continue
            f = replies_with(m, vectors, habit, c["how"], c["layer"], c["alpha"], fact_q, CAP["facts"])
            c["facts"], c["facts_garbage"] = facts_right(fact_pairs, f), sum(map(garbage, f)) / len(f)
            print(habit, "guard check", {k: c[k] for k in ("how", "layer", "alpha", "score", "facts")}, flush=True)
            if c["facts"] >= res["base_facts"] - 0.1 and c["facts_garbage"] <= 0.05:
                chosen = {k: c[k] for k in ("how", "layer", "alpha")}
                break
        res["habits"][habit] = {"rows": rows, "chosen": chosen}
        if chosen:
            unit, size = vectors[(chosen["how"], chosen["layer"])]
            store[habit] = {"unit": unit, "size": size, **chosen}
        print(habit, "CHOSEN", chosen, flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        torch.save(store, str(args.out) + ".vectors.pt")
    res["minutes"] = (time.time() - t0) / 60
    res["tokens"] = m.tokens
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("DEV DONE", res["minutes"], flush=True)


def speed(args) -> None:
    t0 = time.time()
    m = Model(args.threads)
    print("loaded", round(time.time() - t0, 1), "s", flush=True)
    t0 = time.time()
    r = m.write(GENERAL["dev"][:8], 40)
    dt = time.time() - t0
    print(f"{m.tokens} tokens in {dt:.0f} s: {m.tokens / dt:.1f} tokens/s", flush=True)
    print(r[0][:200])
    print(m.write(["Who are you?"], 30))
    print(m.net.config.num_hidden_layers, "layers; hidden size", m.net.config.hidden_size)


def run_test(args) -> None:
    """Test questions, once, with the settings chosen on dev. The rules are in lab/HABITS.md."""
    m = Model(args.threads)
    store = torch.load(str(args.dev) + ".vectors.pt", weights_only=False)
    pairs = FACTS["test"]
    fact_q = [q for q, _ in pairs]
    res: dict = {"base_facts": facts_right(pairs, m.write(fact_q, CAP["facts"])), "habits": {}}
    print("base facts", res["base_facts"], flush=True)
    passed = {}
    for habit, note in HABITS.items():
        qs, cap = set_for(habit, "test"), CAP[habit]
        plain, usual = m.write(qs, cap), m.write(qs, cap, note=note)
        row = {"questions": len(qs), "none": score(habit, plain), "usual": score(habit, usual),
               "usual_facts": facts_right(pairs, m.write(fact_q, CAP["facts"], note=note)),
               "note_tokens": len(m.tok(" " + note)["input_ids"]), "samples_none": plain[:3],
               "samples_usual": usual[:3]}
        passed[habit] = False
        if habit in store:
            s = store[habit]
            steer = (s["layer"], s["alpha"] * s["size"] * s["unit"])
            t0, n0 = time.time(), m.tokens
            vec = m.write(qs, cap, steer=steer)
            speed_now = (m.tokens - n0) / (time.time() - t0)
            f = m.write(fact_q, CAP["facts"], steer=steer)
            mine = score(habit, vec)
            bad = sum(map(garbage, vec + f)) / len(vec + f)
            facts = facts_right(pairs, f)
            bar = {"short": mine >= 0.5 * row["none"], "asks": mine >= row["none"] + 0.3,
                   "admits": mine >= row["none"] + 0.3, "name": mine >= 0.6}[habit]
            guards = facts >= res["base_facts"] - 0.1 and bad <= 0.05
            passed[habit] = bool(bar and guards)
            row.update(vector=mine, vector_facts=facts, vector_garbage=bad, bar=bool(bar), guards=bool(guards),
                       settings={k: s[k] for k in ("how", "layer", "alpha")}, tokens_per_s=speed_now,
                       vector_bytes=int(s["unit"].numel()) * 4, samples_vector=vec[:6])
        res["habits"][habit] = row
        print(habit, {k: v for k, v in row.items() if not k.startswith("samples")}, "PASS" if passed[habit] else "fail",
              flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    res["passed"] = passed
    res["overall"] = passed["short"] and any(passed[h] for h in ("asks", "admits", "name"))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("TEST DONE overall", res["overall"], passed, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["speed", "dev", "test"])
    p.add_argument("--out", default="lab/out/habits_dev.json")
    p.add_argument("--dev", default="lab/out/habits_dev.json")
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    {"speed": speed, "dev": run_dev, "test": run_test}[a.mode](a)


if __name__ == "__main__":
    main()
