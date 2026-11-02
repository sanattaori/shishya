"""Shishya v6: v5's memory, with a childhood in which facts change and many facts share one sentence (docs/V6.md).

    python -m brain.v6 childhood --who first --llm brain:brain/out/runs/brain2 --layer 2 --init <v5 l2 run> --out D
    python -m brain.v6 dev --runs D1 D2 --llm brain:... --out brain/out/v6x/dev.json     # the dev choice

What changes from v5:
- **The childhood's worlds** (train names, v5's 26 train relations): each fact is taught once, in a wording from v5
  or from the teacher (a free LLM, ``brain/teacher_lessons.json``). A third of the people get all their facts in one
  sentence (the teacher's clauses). Half the facts then change 1–3 times, in the teacher's change wordings, after
  all first lessons. The questions ask the current value, in any lesson wording. So the meaning part of the address
  (W) is trained to put "moved to" where "lives in" is asked, and to keep a person's facts apart.
- **Two "who" codes**, chosen on dev: v5's (every new name read so far) or the first new name only.
Everything else is v5: the delta rule, the gate, the memory, no gradients in life.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import v5

TEACHER = Path(__file__).with_name("teacher_lessons.json")


def teacher() -> dict[str, dict[str, list[str]]]:
    return json.loads(TEACHER.read_text())["relations"]


def lesson_wordings(rel: str, T: dict) -> list[str]:
    return list(v5.RELATIONS[rel]) + list(T[rel]["lesson"])


def episode6(rng: random.Random, names: list[str], relations: list[str], T: dict, people: int = 8, per: int = 4,
             p_dense: float = 1 / 3, p_change: float = 0.5, max_changes: int = 3):
    """(lessons in the order read, questions [(prompt, current answer)])."""
    pool = iter(rng.sample(names, people + people * per * (1 + max_changes)))
    first, rounds, questions = [], [[] for _ in range(max_changes)], []
    for _ in range(people):
        s = next(pool)
        rels = rng.sample(relations, per)
        objs = {r: next(pool) for r in rels}
        if rng.random() < p_dense and all(T[r]["clause"] for r in rels):
            cls = [rng.choice(T[r]["clause"]).format(o=objs[r]) for r in rels]
            first.append(f"{s} " + ", ".join(cls[:-1]) + " and " + cls[-1] + ".")
        else:
            first += [rng.choice(lesson_wordings(r, T)).format(s=s, o=objs[r]) for r in rels]
        for r in rels:
            cur = objs[r]
            if T[r]["change"] and rng.random() < p_change:
                for c in range(rng.randint(1, max_changes)):
                    cur = next(pool)
                    rounds[c].append(rng.choice(T[r]["change"]).format(s=s, o=cur))
            q = rng.choice(lesson_wordings(r, T)).format(s=s, o=cur)
            questions.append((q[: q.rindex(" " + cur)], cur))
    rng.shuffle(first)
    for rd in rounds:
        rng.shuffle(rd)
    return first + [x for rd in rounds for x in rd], questions


def dev_episodes(names: list[str], T: dict, n: int = 6):
    rng = random.Random(99)
    return [episode6(rng, names, v5.DEV_RELATIONS, T, people=6, per=4) for _ in range(n)]


def childhood(args) -> None:
    T = teacher()
    v5.childhood(args, make_episode=lambda rng, names: episode6(rng, names, v5.TRAIN_RELATIONS, T),
                 dev_episodes=lambda names: dev_episodes(names, T))


# -- the dev choice (docs/V6.md) -------------------------------------------------------------------------------------
def dev_score(r: dict) -> float:
    """The mean of: hard school A (current value, plain + other words), B plain, B other words, V5's school plain
    and other words (after the quiz). All on dev names."""
    return float(np.mean([r["changes"]["current_both"], r["dense"]["plain"], r["dense"]["other_words"],
                          r["school"]["after_quiz"]["plain"], r["school"]["after_quiz"]["other_words"]]))


def choose(rows: dict[str, dict]) -> str:
    """The highest dev score; a tie goes to v5's who code ("all")."""
    return max(rows, key=lambda k: (round(dev_score(rows[k]), 6), k.endswith("all")))


def dev(args) -> None:
    from .hardschool import changes, dense
    from .school import TemplateGuru, make_world
    rows = {}
    for run in args.runs:
        st, sp, tok, llm = v5.load_student(run, args.llm, args.data_dir, args.device)
        w = make_world(sp.names["dev"], 40, seed=0)
        r = {"changes": changes(st, w, sp.life_dev, sp.names["dev"], 0)}
        st, *_ = v5.load_student(run, args.llm, args.data_dir, args.device)
        r["dense"] = dense(st, w, sp.life_dev, 0)
        st, *_ = v5.load_student(run, args.llm, args.data_dir, args.device)
        r["school"] = v5.run_school(st, w, TemplateGuru(w, sp.life_dev), sp.life_dev)
        for k in ("changes", "dense"):
            r[k] = {a: b for a, b in r[k].items() if not a.startswith("correct")}
        r["score"] = dev_score(r)
        rows[run] = r
        print(run, json.dumps(r), flush=True)
    out = {"rows": rows, "chosen": choose(rows)}
    print("CHOSEN", out["chosen"], flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("childhood")
    c.add_argument("--who", choices=["all", "first"], default="all")
    c.add_argument("--init", help="start from this childhood's reader (a v5 run folder)")
    c.add_argument("--pool", action="store_true", help="v7: the reader also reads the clause's mean state")
    c.add_argument("--layer", type=int, default=2)
    c.add_argument("--r", type=int, default=32)
    c.add_argument("--power", type=float, default=16.0)
    c.add_argument("--steps", type=int, default=600)
    c.add_argument("--check", type=int, default=50)
    c.add_argument("--lr", type=float, default=3e-3)
    c.add_argument("--out", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--runs", nargs="+", required=True)
    d.add_argument("--out")
    for x in (c, d):
        x.add_argument("--llm", default=v5.LLM)
        x.add_argument("--data-dir", default="brain/out/data")
        x.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    (childhood if args.cmd == "childhood" else dev)(args)


if __name__ == "__main__":
    main()
