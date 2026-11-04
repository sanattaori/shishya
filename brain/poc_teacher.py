"""POC (docs/V8_POC.md, amendment 4): a real free AI restates each sentence before the Brain stores it.

The prompts hold none of the exam's wordings, and the teacher never sees the world's facts. A reply that isn't a list of
sentences means the original sentence is stored unchanged. Only network failures are retried, never a reply.

    python -m brain.poc_teacher plain --sentences s.json --out t.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

MODEL = "nvidia/nemotron-3.5-super-vl:free"
PROMPT = """Rewrite the sentence as short, simple sentences for a memory that stores one fact per sentence.
Rules:
- One fact per sentence. Keep every name exactly as written. Add nothing that is not said.
- Keep a fact and its name together: "Mia drives a car named Bolt." stays one sentence.
- State each fact as it is true now, in the most ordinary plain wording, in the present tense. Leave out words about \
change or time, such as now, new, moved, these days.
- Always use the same wording for the same kind of fact, whoever it is about.
Example: "These days Tom has a new job at Acme and Mia, his sister, teaches in Rome." becomes \
["Tom works at Acme.", "Mia is Tom's sister.", "Mia teaches in Rome."]
Reply with only a JSON list of strings.
"""
NOTES = """Already stored about these names:
{notes}
If the sentence gives a new value for a stored fact, copy that stored sentence word for word and change only the value.
"""
NOT_NAMES = {"The", "These", "Now", "Everyone", "Mia", "Tom"}


def names_in(text: str) -> list[str]:
    return [w for w in dict.fromkeys(re.findall(r"[A-Z][a-z]+", text)) if w not in NOT_NAMES]


def ask(client: httpx.Client, prompt: str) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    last = ""
    for attempt in range(12):
        try:
            r = client.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                            json={"model": MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200 and r.json().get("choices"):
                return r.json()["choices"][0]["message"]["content"] or ""
            last = f"{r.status_code} {r.text[:120]}"
        except (httpx.HTTPError, ValueError) as e:
            last = type(e).__name__
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"the free model did not answer: {last}")


def parse(reply: str, sentence: str) -> tuple[list[str], bool]:
    """The teacher's sentences, and whether the reply could be read."""
    m = re.search(r"\[.*\]", reply, re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except ValueError:
        out = None
    if isinstance(out, list) and out and all(isinstance(x, str) and x.strip() for x in out):
        return [x.strip() for x in out], True
    return [sentence], False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teacher", choices=["plain", "notebook"])
    p.add_argument("--sentences", required=True, help="a JSON list of the lessons, in order")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    lessons = json.load(open(a.sentences))
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    client = httpx.Client(timeout=120)
    t0 = time.time()

    def save():
        tmp = a.out + ".tmp"
        json.dump(done, open(tmp, "w"), indent=0)
        os.replace(tmp, a.out)

    if a.teacher == "plain":
        todo = [s for s in lessons if s not in done]

        def one(s):
            return s, ask(client, PROMPT + "Sentence: " + s)

        with ThreadPoolExecutor(a.workers) as ex:
            for s, reply in ex.map(one, todo):
                done[s] = {"out": parse(reply, s)[0], "read": parse(reply, s)[1], "reply": reply}
                save()
    else:
        book: dict[str, list[str]] = {}
        for s in lessons:                                   # in order: each lesson sees only what came before it
            if s not in done:
                known = list(dict.fromkeys(x for n in names_in(s) for x in book.get(n, [])))
                extra = NOTES.format(notes="\n".join(f"- {x}" for x in known)) if known else ""
                reply = ask(client, PROMPT + extra + "Sentence: " + s)
                done[s] = {"out": parse(reply, s)[0], "read": parse(reply, s)[1], "reply": reply}
                save()
            for x in done[s]["out"]:
                for n in names_in(x):
                    book.setdefault(n, []).append(x)
    save()
    print(f"DONE {len(done)} sentences, unreadable {sum(not v['read'] for v in done.values())}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
