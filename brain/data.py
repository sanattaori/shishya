"""TinyStories as bytes, made-up facts, childhood episodes and the life stream.

Names are made-up words (``Zubak``, ``Morif``), split once and for all into train / dev / test lists by a hash.
Childhood drills use only train names; the dev stream (choosing learning rates) uses dev names; the test stream
uses test names. Names that occur in the loaded stories are dropped, so no model knows them in advance.
"""
from __future__ import annotations

import os
import random
import re
import shutil
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
SEP = "<|endoftext|>"
SESSION_BYTES = 256

CONS = "bdfgkmnprstvz"
VOWS = "aeiou"
# a few very common TinyStories names and words, dropped even before the stories are checked
COMMON = {"timmy", "sara", "lily", "momo", "bobo", "tom", "ben", "sam", "max", "mia", "sue", "kate", "pip"}

RELATIONS = [
    # (fact sentence, question prompt, the same question phrased in a way never seen in childhood)
    ("{s}'s {pet} is called {o}.", "{s}'s {pet} is called", "The {pet} of {s} is called"),
    ("{s} lives in a town called {o}.", "{s} lives in a town called", "The town where {s} lives is called"),
    ("{s}'s best friend is {o}.", "{s}'s best friend is", "The best friend of {s} is"),
    ("{s} has a toy named {o}.", "{s} has a toy named", "The toy that {s} has is named"),
    ("The secret word of {s} is {o}.", "The secret word of {s} is", "{s}'s secret word is"),
]
PETS = ["cat", "dog", "bird", "fish", "frog", "duck", "bunny"]

FULL_LIFE = {"main": 100, "early": 20, "once": 50, "reward": 50, "change": 30, "per_session": 2}
SMALL_LIFE = {"main": 12, "early": 4, "once": 6, "reward": 6, "change": 4, "per_session": 2}
REWARD = 3.0


# -- names ----------------------------------------------------------------------------------------------------------
def _all_names() -> list[str]:
    out = []
    for a in CONS:
        for b in VOWS:
            for c in CONS:
                for d in VOWS:
                    for e in CONS:
                        out.append((a + b + c + d + e).capitalize())       # 5 letters: CVCVC, e.g. Zubak
    return out


def name_split(exclude: set[str] | None = None) -> dict[str, list[str]]:
    """train 70%, dev 10%, test 20% by crc32; names in ``exclude`` (lower case) or COMMON are dropped."""
    bad = COMMON | (exclude or set())
    out = {"train": [], "dev": [], "test": []}
    for n in _all_names():
        if n.lower() in bad:
            continue
        h = zlib.crc32(n.encode()) % 10
        out["train" if h < 7 else "dev" if h == 7 else "test"].append(n)
    return out


# -- stories --------------------------------------------------------------------------------------------------------
def fetch(data_dir: str | Path, split: str, max_bytes: int | None = None) -> Path:
    """Download TinyStoriesV2 (GPT-4 part). ``max_bytes`` downloads only the start of the file."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"TinyStoriesV2-GPT4-{split}.txt"
    if path.exists() and (max_bytes is None or path.stat().st_size >= max_bytes * 0.99):
        return path
    req = urllib.request.Request(URL.format(split=split))
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)
    return path


def load_stories(path: str | Path, limit_bytes: int | None = None) -> list[str]:
    with open(path, "rb") as f:
        raw = f.read(limit_bytes) if limit_bytes else f.read()
    text = raw.decode("utf-8", errors="ignore")
    parts = [p.strip() for p in text.split(SEP)]
    if limit_bytes:
        parts = parts[:-1]                                                  # the last one may be cut
    return [p for p in parts if len(p) > 40]


def _bucket(s: str) -> int:
    return zlib.crc32(s.encode()) % 10


@dataclass
class Splits:
    train: list[str]
    val: list[str]                 # held-out bits per byte
    life_dev: list[str]
    life_test: list[str]
    names: dict[str, list[str]]
    smoke: bool


def splits(data_dir: str | Path, smoke: bool = False, train_bytes: int = 400_000_000) -> Splits:
    """The validation file gives val / life_dev / life_test (by hash). Train comes from the train file, or in
    ``smoke`` mode from the rest of the validation file (the CPU check only)."""
    valid = load_stories(fetch(data_dir, "valid"))
    val = [s for s in valid if _bucket(s) < 2]
    life_dev = [s for s in valid if _bucket(s) in (2, 3)]
    life_test = [s for s in valid if _bucket(s) in (4, 5)]
    if smoke:
        train = [s for s in valid if _bucket(s) >= 6]
    else:
        train = load_stories(fetch(data_dir, "train", train_bytes), limit_bytes=train_bytes)
    words = set()
    for s in valid + train[:20000]:
        words.update(re.findall(r"[a-z]+", s.lower()))
    return Splits(train, val, life_dev, life_test, name_split(words), smoke)


def encode(text: str, T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(inputs [T], targets [T], mask [T]) from the first T+1 bytes; short texts are padded with 0, masked out."""
    b = text.encode("utf-8")[: T + 1]
    arr = np.zeros(T + 1, dtype=np.int64)
    arr[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    mask = np.zeros(T, dtype=bool)
    mask[: max(len(b) - 1, 0)] = True
    return arr[:-1], arr[1:], mask


def _fit(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` bytes without breaking a character."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _sentences(story: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", story) if s]


def with_facts(story: str, facts: list[str], limit: int) -> str:
    """The story's first sentence, then the fact sentences, then the rest; at most ``limit`` bytes and the facts
    always fully inside."""
    sents = _sentences(story)
    head = sents[0] if sents and len(sents[0].encode()) < limit // 3 else ""
    core = " ".join(([head] if head else []) + facts)
    rest = " ".join(sents[1:] if head else sents)
    room = limit - len(core.encode()) - 1
    return core + (" " + _fit(rest, room) if room > 0 else "")


# -- facts ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Fact:
    subject: str
    relation: int
    pet: str
    obj: str

    def _fill(self, t: str) -> str:
        return t.format(s=self.subject, pet=self.pet, o=self.obj)

    @property
    def sentence(self) -> str:
        return self._fill(RELATIONS[self.relation][0])

    @property
    def prompt(self) -> str:
        return self._fill(RELATIONS[self.relation][1])

    @property
    def paraphrase(self) -> str:
        return self._fill(RELATIONS[self.relation][2])


def make_fact(rng: random.Random, subject: str, obj: str) -> Fact:
    return Fact(subject, rng.randrange(len(RELATIONS)), rng.choice(PETS), obj)


# -- childhood ------------------------------------------------------------------------------------------------------
def episode(rng: random.Random, stories: list[str], names: list[str], T: int, S: int, p_drill: float) -> list[str]:
    """S segment texts. A story episode splits one story across the first and last segment, with other stories
    between. A drill puts 1–3 made-up facts in the first segment and states them again in the last."""
    others = [rng.choice(stories) for _ in range(S - 2)]
    if rng.random() < p_drill:
        k = rng.randint(1, 3)
        picked = rng.sample(names, 2 * k)
        facts = [make_fact(rng, picked[2 * i], picked[2 * i + 1]) for i in range(k)]
        first = with_facts(rng.choice(stories), [f.sentence for f in facts], T + 1)
        rng.shuffle(facts)
        last = " ".join(f.sentence for f in facts) + " " + rng.choice(stories)
        return [first] + others + [_fit(last, T + 1)]
    story = rng.choice(stories)
    sents = _sentences(story)
    cut = max(1, len(sents) // 2)
    return [" ".join(sents[:cut])] + others + [" ".join(sents[cut:]) or story]


def childhood_batch(rng: random.Random, stories, names, B: int, T: int, S: int, p_drill: float):
    """Arrays [B, S, T] of inputs, targets and mask."""
    xs, ys, ms = (np.zeros((B, S, T), dtype=t) for t in (np.int64, np.int64, bool))
    for b in range(B):
        for s, text in enumerate(episode(rng, stories, names, T, S, p_drill)):
            xs[b, s], ys[b, s], ms[b, s] = encode(text, T)
    return xs, ys, ms


# -- the life stream ------------------------------------------------------------------------------------------------
@dataclass
class Session:
    text: str
    facts: list[str] = field(default_factory=list)       # fact sentences in this session
    reward: float = 1.0


@dataclass
class LifeStream:
    sessions: list[Session]
    facts: dict[str, list[Fact]]                         # main, once, reward, change_old, change_new
    early: list[Fact]                                    # main facts fully taught before the checkpoint
    checkpoint: int                                      # evaluate the early facts after this many sessions
    candidates: dict[str, list[str]]                     # fact sentence -> 5 objects, the right one first


def life_stream(stories: list[str], names: list[str], seed: int, sizes: dict = FULL_LIFE) -> LifeStream:
    rng = random.Random(seed)
    pool = list(names)
    rng.shuffle(pool)
    need = 2 * (sizes["main"] + sizes["once"] + sizes["reward"]) + 3 * sizes["change"]
    assert len(pool) >= need, "not enough names"
    take = iter(pool)
    facts = {k: [make_fact(rng, next(take), next(take)) for _ in range(sizes[k])] for k in ("main", "once", "reward")}
    facts["change_old"], facts["change_new"] = [], []
    for _ in range(sizes["change"]):
        old = make_fact(rng, next(take), next(take))
        facts["change_old"].append(old)
        facts["change_new"].append(Fact(old.subject, old.relation, old.pet, next(take)))
    early = facts["main"][: sizes["early"]]

    # events: (fact, reward). Corrections (reward and changes) get a session of their own.
    early_ev = [(f, 1.0) for f in early for _ in range(3)]
    rest_ev = ([(f, 1.0) for f in facts["main"][sizes["early"]:] for _ in range(3)]
               + [(f, 1.0) for f in facts["once"]] + [(f, REWARD) for f in facts["reward"]]
               + [(f, 1.0) for f in facts["change_old"] for _ in range(3)])
    rng.shuffle(early_ev)
    rng.shuffle(rest_ev)
    n_early_rest = max(0, int(0.3 * (len(early_ev) + len(rest_ev))) - len(early_ev))
    first = early_ev + rest_ev[:n_early_rest]
    rng.shuffle(first)
    events = first + rest_ev[n_early_rest:]
    for old, new in zip(facts["change_old"], facts["change_new"]):        # each change after its last old exposure
        last = max(i for i, (f, _) in enumerate(events) if f == old)
        events.insert(rng.randint(last + 1, len(events)), (new, REWARD))

    sessions, story_i = [], 0
    i = 0

    def story():
        nonlocal story_i
        story_i += 1
        return stories[(story_i - 1) % len(stories)]

    while i < len(events):
        f, r = events[i]
        group = [events[i]]
        i += 1
        if r == 1.0:
            while (len(group) < sizes["per_session"] and i < len(events) and events[i][1] == 1.0
                   and events[i][0] not in [g[0] for g in group]):
                group.append(events[i])
                i += 1
        sents = [g[0].sentence for g in group]
        sessions.append(Session(with_facts(story(), sents, SESSION_BYTES), sents, group[0][1]))
        sessions.append(Session(_fit(story(), SESSION_BYTES)))             # a plain story between
    early_s = {f.sentence for f in early}
    checkpoint = 2 + max(n for n, s in enumerate(sessions) if early_s & set(s.facts))   # after its plain story

    objs = sorted({f.obj for fs in facts.values() for f in fs})
    candidates = {}
    for kind, fs in facts.items():
        for j, f in enumerate(fs):
            wrong = [o for o in objs if o != f.obj]
            r2 = random.Random(f"{seed}:{f.sentence}")
            pick = r2.sample(wrong, 4)
            if kind == "change_new":                                       # the old answer is always a choice
                old = facts["change_old"][j].obj
                pick = [old] + [o for o in pick if o != old][:3]
            candidates[f.sentence] = [f.obj] + pick
    return LifeStream(sessions, facts, early, checkpoint, candidates)
