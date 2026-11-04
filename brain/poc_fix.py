"""POC (docs/V8_POC.md, amendment 6): hide replaced values from the sharp read. The table is not touched.

    python -m brain.poc_fix changes.pt dense.pt --teach-changes t.json --teach-dense t.json --rule same_words
"""
from __future__ import annotations

import argparse
import json
import re

import torch

from . import poc_table as T
from .poc_diag import fact_names
from .poc_teacher import names_in

SHARP = dict(power=4, near=0.9, gain=4, newest=False)


def key_value(x: str) -> tuple[str, str] | None:
    """A sentence without its last name, and that name. None if no other name is left to say who it is about."""
    names = names_in(x)
    if len(names) < 2:
        return None
    last = list(re.finditer(r"[A-Z][a-z]+", x))
    last = [m for m in last if m.group(0) in names][-1]
    return x[:last.start()] + "_" + x[last.end():], last.group(0)


def replaced_same_words(lessons: list[str], teach: dict) -> set[str]:
    """Values replaced by a later sentence that differs only in its last name. Uses no knowledge of the world."""
    now: dict[str, str] = {}
    old: set[str] = set()
    for s in lessons:
        for x in teach[s]["out"]:
            kv = key_value(x)
            if kv is None:
                continue
            k, v = kv
            if k in now and now[k] != v:
                old.add(now[k])
            now[k] = v
    return old


def old_in_exam(d: dict) -> set[str]:
    """The ceiling: the exam's own list of old values. Option 0 is right; 1 to 3 are the old ones."""
    items = d["exams"][0]
    return {items[q + k]["cont"].strip(" .") for q in range(0, len(items), 5) for k in (1, 2, 3)}


def hide(d: dict, rows_all: list[list[dict]], values: set[str]) -> tuple[int, int]:
    names = fact_names(d)
    facts = {f for f, (_, n) in names.items() if n in values}
    mask = torch.tensor([f in facts for f in d["fact"].tolist()])
    for rows in rows_all:
        for r in rows:
            if r["s"] is not None:
                r["s"] = r["s"].clone()
                r["s"][:, mask] = 0
    return len(facts), int(mask.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("changes")
    p.add_argument("dense")
    p.add_argument("--teach-changes", required=True)
    p.add_argument("--rule", choices=["none", "ceiling", "same_words"], required=True)
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(2)
    dc, trc, exc, tbc = T.load(a.changes, None)
    dd, trd, exd, tbd = T.load(a.dense, None)
    if a.rule == "ceiling":
        vals = old_in_exam(dc)
    elif a.rule == "same_words":
        teach = json.load(open(a.teach_changes))
        vals = replaced_same_words(list(teach), teach)
    else:
        vals = set()
    n = hide(dc, exc, vals)
    r = T.evaluate(dc, trc, exc, tbc, dd, trd, exd, tbd, 1.0, SHARP)
    print(a.rule, "values hidden", len(vals), "facts, traces", n,
          {k: round(r[k] * 100, 1) for k in ("changes", "plain", "other", "kept", "dense", "dense_other")}, flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
