"""Ask an outside LLM (OpenRouter) to write exam questions for the five test relations, in its own words.

    set -a; . ./.env; set +a; .venv/bin/python -m brain.notes.guru_questions --out brain/guru_questions.json

Each wording must name what it asks about (the pet kind, "town", "friend", "toy", "secret"), so a person
could answer it. Evaluation only: these wordings are never used in childhood or in any lesson. Each is a sentence
start whose next word is the answer (a made-up name), with {s} for the person and {p} for the pet kind.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

from brain import data
from brain.school import ASK, TWO_STEP, WORDINGS

MODELS = ["google/gemma-4-31b-it:free", "nvidia/nemotron-3-ultra-550b-a55b:free"]   # free on OpenRouter
MODEL = MODELS[0]
KEYWORD = {"pet": "{p}", "town": "town", "friend": "friend", "toy": "toy", "secret": "secret"}
WHAT = {"pet": "the name of {s}'s pet (a {p}, for example a cat or a dog)",
        "town": "the name of the town where {s} lives",
        "friend": "the name of {s}'s best friend", "toy": "the name of {s}'s toy",
        "secret": "{s}'s secret word"}


def known_wordings(rel: str) -> set[str]:
    i = list(WORDINGS).index(rel)
    out = {w for w in WORDINGS[rel][0]} | set(WORDINGS[rel][1]) | {ASK[rel]} | set(TWO_STEP.values())
    out |= {data.RELATIONS[i][0], data.RELATIONS[i][1], data.RELATIONS[i][2], data.HELDOUT_PARAPHRASE[i]}
    return out


def ask(rel: str, n: int) -> list[str]:
    avoid = "\n".join(sorted(known_wordings(rel)))
    prompt = f"""Write {n} different ways to start a sentence so that the very next word is {WHAT[rel]}.
Rules:
- Use {{s}} for the person's name{" and {p} for the kind of pet" if rel == "pet" else ""}. Each line must contain {{s}}.
- The sentence must stop right before the answer: the answer is not written. No question marks.
- Each line must say what is being named, using the word "{KEYWORD[rel]}", so a reader can tell what is asked.
- Simple English, as in a children's story. Vary the grammar and the words a lot.
- Do not reuse these wordings or near copies of them:
{avoid}
Answer with only the {n} lines, one per line, no numbers."""
    global MODEL
    text = None
    for attempt in range(12):                     # free models are rate-limited: wait, then try the next one
        MODEL = MODELS[attempt % len(MODELS)]
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
                            ).encode(),
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as e:
            print("  ", MODEL, e.code, "- waiting")
            time.sleep(20)
    assert text, "no free model answered"
    return [ln.strip().strip("-").strip() for ln in text.splitlines() if ln.strip()]


def valid(rel: str, w: str) -> bool:
    if KEYWORD[rel] not in w.lower() and KEYWORD[rel] not in w:
        return False
    if "{s}" not in w or w.count("{s}") != 1 or "?" in w or w.endswith((".", "!")):
        return False
    if re.search(r"\{(?!s\}|p\})", w) or ("{p}" in w and rel != "pet"):
        return False
    norm = lambda x: re.sub(r"\W+", " ", x.lower()).strip()  # noqa: E731
    return norm(w) not in {norm(k) for k in known_wordings(rel)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default="brain/guru_questions.json")
    args = p.parse_args()
    out = {}
    for rel in WORDINGS:
        got = [w.rstrip(" ,:;") for w in ask(rel, args.n + 8)]
        ok = [w for w in dict.fromkeys(got) if valid(rel, w)][: args.n]
        out[rel] = ok
        print(rel, MODEL, len(got), "->", len(ok))
    json.dump({"models": MODELS, "wordings": out}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
