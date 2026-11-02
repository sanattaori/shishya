"""The teacher (v6): a free LLM on OpenRouter writes how to teach each kind of fact, in its own words.

    set -a; . ./.env; set +a; .venv/bin/python -m brain.notes.teacher --out brain/teacher_lessons.json

For every relation (v5's 32 childhood relations and the five test relations) it writes:
- lesson: sentences that state the fact ("{s}'s teacher is {o}.")
- change: sentences that say the fact has changed to a new value ("{s} moved to {o}.")
- clause: parts of a longer sentence about {s} ("has a teacher called {o}"), to put many facts in one sentence

The teacher's models are not the examiner's (brain/guru_questions.json: Gemma and Nemotron), and for the five test
relations nothing that copies an exam wording, an examiner wording or the hard school's change wordings is kept.
The answer is cached here and committed, so every run uses the same lessons.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

from brain import v5
from brain.hardschool import CHANGE
from brain.notes.guru_questions import WHAT, known_wordings
from brain.school import WORDINGS

MODELS = ["dots-studio/dots-3-note-preview:free", "xingchen-agi/xing4.0-29b-a4b:free"]   # free; not the examiner's
TEST = list(WORDINGS)                                     # pet, town, friend, toy, secret
N = {"lesson": 8, "change": 6, "clause": 4}


def what(rel: str) -> str:
    return WHAT[rel] if rel in WHAT else f"the name of {{s}}'s {rel}"


def keyword(rel: str) -> str:
    return {"pet": "{p}", "friend": "friend", "secret": "secret"}.get(rel, rel)


def prompt(rel: str) -> str:
    pet = " Use {p} for the kind of pet (cat, dog, ...)." if rel == "pet" else ""
    return f"""You are a teacher writing simple sentences for a children's story. The fact to teach is {what(rel)}.
Use {{s}} for the person and {{o}} for the name being taught (a made-up name).{pet}

Write a JSON object with three lists:
- "lesson": {N["lesson"]} different sentences that state the fact. Each contains {{s}} once, uses the word
  "{keyword(rel)}", and ends with {{o}} followed by a full stop.
- "change": {N["change"]} different sentences saying the fact has CHANGED and the new one is {{o}} (for example
  that {{s}} moved, got a new one, or now has another). Each contains {{s}} once and ends with {{o}} and a full stop.
- "clause": {N["clause"]} short verb phrases about {{s}} without {{s}} in them, each ending with {{o}} and no full stop
  (for example "has a ... called {{o}}"), so several can be joined: "{{s}} <clause>, <clause> and <clause>."
Vary the grammar and the words a lot. Answer with only the JSON object."""


def ask(rel: str) -> tuple[dict, str]:
    for attempt in range(16):                     # free models are rate-limited: wait, then try the other one
        model = MODELS[attempt % len(MODELS)]
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt(rel)}],
                             "temperature": 0.8}).encode(),
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = json.loads(r.read())["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)), model
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, AttributeError, ValueError, KeyError,
                TypeError) as e:
            print("  ", model, type(e).__name__, getattr(e, "code", ""), "- waiting", flush=True)
            time.sleep(15)
    raise RuntimeError(f"no free model answered for {rel}")


def _norm(x: str) -> str:
    return re.sub(r"\W+", " ", x.lower().replace("{o}", "")).strip()


def banned(rel: str) -> set[str]:
    """For a test relation: its exam, examiner and hard-school wordings (as sentence starts)."""
    if rel not in TEST:
        return set()
    out = {_norm(w) for w in known_wordings(rel)}
    gw = v5.guru_wordings() or {}
    out |= {_norm(w) for w in gw.get(rel, [])}
    out |= {_norm(w) for w in CHANGE.get(rel, [])}
    return out


def valid(rel: str, kind: str, w: str, ban: set[str]) -> bool:
    w = w.strip()
    allowed = {"{s}", "{o}", "{p}"} if rel == "pet" else {"{s}", "{o}"}
    if set(re.findall(r"\{[^}]*\}", w)) - allowed or w.count("{o}") != 1:
        return False
    if kind == "clause":
        if "{s}" in w or not w.endswith("{o}"):
            return False
    elif w.count("{s}") != 1 or not w.endswith("{o}."):
        return False
    if kind == "lesson" and keyword(rel).lower() not in w.lower():
        return False
    return _norm(w) not in ban


def teach(rel: str) -> tuple[dict, str]:
    ban = banned(rel)
    keep: dict[str, list[str]] = {k: [] for k in N}
    model = ""
    for attempt in range(4):                      # ask again while a list is short (merging the new lines)
        got, model = ask(rel)
        for k in N:
            for w in got.get(k, []) if isinstance(got.get(k), list) else []:
                if isinstance(w, str) and valid(rel, k, w, ban) and w.strip() not in keep[k]:
                    keep[k].append(w.strip())
        print("  ", rel, model, attempt, {k: len(v) for k, v in keep.items()}, flush=True)
        if all(len(keep[k]) >= N[k] - 2 for k in N):
            break
    return {k: v[: N[k]] for k, v in keep.items()}, model


def main():
    from concurrent.futures import ThreadPoolExecutor
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="brain/teacher_lessons.json")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    out, used = {}, {}
    if os.path.exists(args.out):                  # resume
        prev = json.load(open(args.out))
        out, used = prev["relations"], prev["model_per_relation"]
    todo = [r for r in list(v5.RELATIONS) + TEST if r not in out]
    with ThreadPoolExecutor(args.workers) as ex:
        for rel, (lists, model) in zip(todo, ex.map(teach, todo)):
            out[rel], used[rel] = lists, model
            print(rel, model, {k: len(v) for k, v in lists.items()}, flush=True)
            order = [r for r in list(v5.RELATIONS) + TEST if r in out]
            json.dump({"models": MODELS, "model_per_relation": {r: used[r] for r in order},
                       "relations": {r: out[r] for r in order}}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
