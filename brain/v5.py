"""Shishya v5: a memory that keeps meaning (who ⊗ what), on a frozen LLM.

The school (brain/RESULTS.md section 9) and the hybrid (docs/HYBRID.md) hit the same wall: the memory recalls a fact
asked in the words it was read in, but not in other words. The plan, rules and results are in docs/V5.md.

**The address has two parts.**

- **who** picks the memory's units. Shishya's word code (v3) for the words of the sentence so far, each counting by
  how new it is raised to ``power`` (16), so only new names count. The strongest 40 of 16,384 units are kept.
  The same names give the same units, in any wording.
- **what** is a small dense vector (32 numbers): the frozen LLM's hidden state at a middle layer, over the sentence
  alone, through a projection ``W`` learned once in childhood. It says what the sentence is about to name.

**The memory** holds, in each unit, a 32 × 128 matrix: what → value. A fact is stored in the units of its names, at
its meaning. The value is the next token (its embedding through a fixed random projection to 128 numbers). Reading
with a question: the units of its names, read at its meaning. Two addresses match by (who overlap) × (what
cosine): smooth, with no top-k step on the meaning.

**The gate.** A step is written and read only when the sentence so far has a new word (novelty ** power ≥ 0.5).
Ordinary text is never written, so the LLM's reading of it is untouched.

**Life** is the same as v3's: no gradients, the exact sequential delta rule, β = min(rate × surprise × reward, 1).
**Childhood** learns only ``W`` (576 × 32), the write rate and the read gain, with the LLM frozen, on made-up worlds
of 30 relation types (teacher, boat, school, …) that are **not** the five the tests ask about (pet, town, best
friend, toy, secret word).

    python -m brain.v5 childhood --layer 8 --out brain/out/v5/l8      # learn W (train names, train relations)
    python -m brain.v5 school --run brain/out/v5/l8 --names dev       # the school exam (dev names)
    python -m brain.v5 life --run brain/out/v5/l8 --stream dev        # the life test (dev stream)
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .hybrid import BOS, CAP, RAG, LLMLearner, Tokens, load_llm, loss_ref, make_address, run_llm
from .lifetest import Lookup, live
from .models import MAX_WORD, NOVELTY_BUCKETS, Brain

LLM = "HuggingFaceTB/SmolLM2-135M"
_STOP = {ord("."), ord("!"), ord("?"), ord("\n")}
_SPACE = {ord(" "), ord("\t"), ord("\r")}

# Childhood relations: none is one of the five the tests ask about (pet, town, best friend, toy, secret word), and
# none is a near copy of them (no animals, places to live, friends, toys or secret words). The object is always last.
RELATIONS = {
    "teacher": ["{s}'s teacher is {o}.", "The teacher of {s} is {o}.", "{s} has a teacher called {o}.",
                "Every day {s} learns from a teacher named {o}."],
    "sister": ["{s}'s sister is {o}.", "The sister of {s} is called {o}.", "{s} has a little sister named {o}.",
               "The girl who is {s}'s sister is {o}."],
    "brother": ["{s}'s brother is {o}.", "The brother of {s} is named {o}.", "{s} has a big brother called {o}.",
                "{s} grew up with a brother named {o}."],
    "cousin": ["{s}'s cousin is {o}.", "The cousin of {s} is called {o}.", "{s} visits a cousin named {o}.",
               "Once a year {s} sees a cousin called {o}."],
    "uncle": ["{s}'s uncle is {o}.", "The uncle of {s} is named {o}.", "{s} has a funny uncle called {o}.",
              "The man who is {s}'s uncle is {o}."],
    "aunt": ["{s}'s aunt is {o}.", "The aunt of {s} is called {o}.", "{s} bakes cookies with an aunt named {o}.",
             "{s} has a kind aunt called {o}."],
    "grandma": ["{s}'s grandma is {o}.", "The grandma of {s} is named {o}.", "{s} loves a grandma called {o}.",
                "{s} hugs grandma {o}."],
    "doctor": ["{s}'s doctor is {o}.", "The doctor who helps {s} is {o}.", "{s} goes to a doctor named {o}.",
               "When {s} is sick, the doctor is {o}."],
    "coach": ["{s}'s coach is {o}.", "The coach of {s} is called {o}.", "{s} trains with a coach named {o}.",
              "{s} runs fast for coach {o}."],
    "neighbour": ["{s}'s neighbour is {o}.", "The neighbour of {s} is named {o}.", "Next door to {s} lives {o}.",
                  "{s} waves to a neighbour called {o}."],
    "baby": ["{s}'s baby is named {o}.", "The baby of {s} is called {o}.", "{s} sings to a baby named {o}.",
             "{s} has a new baby called {o}."],
    "boss": ["{s}'s boss is {o}.", "The boss of {s} is called {o}.", "{s} works for a boss named {o}.",
             "At work {s} listens to {o}."],
    "boat": ["{s}'s boat is called {o}.", "The boat that {s} sails is named {o}.", "{s} has a small boat called {o}.",
             "{s} goes fishing in a boat named {o}."],
    "ship": ["{s}'s ship is named {o}.", "The ship of {s} is called {o}.", "{s} is the captain of a ship called {o}.",
             "{s} crossed the sea on the ship {o}."],
    "car": ["{s}'s car is called {o}.", "The car {s} drives is named {o}.", "{s} has a red car named {o}.",
            "{s} goes to work in a car called {o}."],
    "song": ["{s}'s favourite song is {o}.", "The song {s} sings most is {o}.", "{s} loves to sing a song called {o}.",
             "{s} hums a tune named {o}."],
    "book": ["{s}'s favourite book is {o}.", "The book {s} reads at night is {o}.", "{s} is reading a book called {o}.",
             "{s} wrote a book named {o}."],
    "poem": ["{s}'s poem is called {o}.", "The poem {s} wrote is named {o}.", "{s} reads out a poem called {o}.",
             "{s} learned a poem named {o}."],
    "school": ["{s}'s school is {o}.", "The school of {s} is named {o}.", "{s} goes to a school called {o}.",
               "Every morning {s} walks to the school {o}."],
    "shop": ["{s}'s shop is called {o}.", "The shop {s} owns is named {o}.", "{s} works at a shop called {o}.",
             "{s} sells bread at the shop {o}."],
    "island": ["{s} was born on an island called {o}.", "The island {s} comes from is {o}.",
               "{s}'s island is named {o}.", "{s} sailed home to the island {o}."],
    "river": ["{s} swims in a river called {o}.", "The river near {s} is named {o}.", "{s}'s river is {o}.",
              "{s} fishes in the river {o}."],
    "mountain": ["{s} climbed a mountain called {o}.", "The mountain {s} sees is named {o}.",
                 "{s}'s mountain is {o}.", "{s} wants to climb the mountain {o}."],
    "star": ["{s}'s star is called {o}.", "The star {s} wishes on is named {o}.", "{s} found a star called {o}.",
             "At night {s} looks for the star {o}."],
    "castle": ["{s}'s castle is called {o}.", "The castle of {s} is named {o}.", "{s} lives like a king in {o}.",
               "{s} built a castle named {o}."],
    "team": ["{s}'s team is {o}.", "The team {s} plays for is called {o}.", "{s} is on a team named {o}.",
             "{s} cheers for the team {o}."],
    "club": ["{s}'s club is {o}.", "The club {s} joined is called {o}.", "{s} meets friends at a club named {o}.",
             "{s} is the leader of the club {o}."],
    "garden": ["{s}'s garden is called {o}.", "The garden of {s} is named {o}.",
               "{s} grows roses in a garden called {o}.", "{s} waters the garden {o}."],
    "hat": ["{s}'s hat is called {o}.", "The hat {s} wears is named {o}.", "{s} has a tall hat called {o}.",
            "{s} never goes out without the hat {o}."],
    "spell": ["{s}'s spell is {o}.", "The spell {s} casts is called {o}.", "{s} knows a magic spell named {o}.",
              "{s} waves a wand and says the spell {o}."],
    "planet": ["{s}'s planet is {o}.", "The planet {s} visits is named {o}.",
               "{s} flew a rocket to a planet called {o}.", "{s} dreams of the planet {o}."],
    "dance": ["{s}'s dance is called {o}.", "The dance {s} does is named {o}.", "{s} invented a dance called {o}.",
              "{s} teaches the dance {o}."],
}
DEV_RELATIONS = ["spell", "planet", "dance", "club", "uncle", "river"]       # held out of childhood: for its checks
TRAIN_RELATIONS = [r for r in RELATIONS if r not in DEV_RELATIONS]


def llm_on(name: str, device: str = "cpu"):
    """The frozen reader: an LLM (float32 on the CPU, float16 on a GPU), or with ``brain:<run folder>`` Shishya's
    own BDH Brain (no LLM at all)."""
    if name.startswith("brain:"):
        from .lifetest import load
        return ByteTok(), BrainLM(load(Path(name[6:]), torch.device(device)))
    if name.startswith("bdh:"):
        return ByteTok(), BDHLM(load_bdh(name[4:])).to(device)
    tok, llm = load_llm(name)
    if device != "cpu":
        llm = llm.half().to(device)
    return tok, llm


class ByteTok:
    """Bytes as tokens, for the Brain (it reads bytes). Offsets are in characters, as a Hugging Face tokenizer's."""

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids, offs = [], []
        for i, ch in enumerate(text):
            for b in ch.encode():
                ids.append(b)
                offs.append((i, i + 1))
        return {"input_ids": ids, "offset_mapping": offs}


class BrainLM(torch.nn.Module):
    """Shishya's BDH Brain, frozen and without its own lasting memory, with the interface v5 uses: logits and the
    hidden state after each layer. The Brain reads text from its first byte, so the BOS put first is dropped and
    position 0 gets uniform logits and a zero state."""

    def __init__(self, brain: Brain):
        super().__init__()
        from types import SimpleNamespace
        self.brain = brain.eval()
        self.config = SimpleNamespace(hidden_size=brain.cfg.n_embd, num_hidden_layers=brain.cfg.n_layer)

    def get_output_embeddings(self):
        from types import SimpleNamespace
        return SimpleNamespace(weight=self.brain.lm_head.T)

    @torch.no_grad()
    def _run(self, idx: torch.Tensor):
        from .models import _ln, _rope
        b, c = self.brain, self.brain.cfg
        B, T = idx.shape
        D, nh, N = c.n_embd, c.n_head, c.neurons
        x = _ln(b.embed(idx).unsqueeze(1))
        phases = torch.arange(T, device=idx.device, dtype=torch.float32).view(1, 1, -1, 1) * b.freqs
        hs = [x.view(B, T, D)]
        for _ in range(c.n_layer):                          # Brain.forward with no lasting memory, in eval mode
            x_sparse = F.relu(x @ b.encoder)
            qr = _rope(phases, x_sparse)
            yKV = ((qr @ qr.mT) * (1.0 / N)).tril(diagonal=-1) @ x
            yKV = _ln(yKV, eps=1e-5 / N**2)
            y_sparse = F.relu(yKV @ b.encoder_v)
            yMLP = (x_sparse * y_sparse).transpose(1, 2).reshape(B, 1, T, N * nh) @ b.decoder
            x = _ln(x + _ln(yMLP))
            hs.append(x.view(B, T, D))
        return x.view(B, T, D) @ b.lm_head, hs

    def forward(self, input_ids, attention_mask=None, output_hidden_states: bool = False):
        from types import SimpleNamespace
        idx = input_ids[:, 1:]
        lgs, hss = [], []
        for i in range(0, idx.size(0), 8):                  # BDH's attention is large: 8 texts at a time
            lg, hs = self._run(idx[i:i + 8])
            lgs.append(lg)
            hss.append(hs)
        lg = torch.cat(lgs)
        B, _, V = lg.shape
        z = lg.new_zeros(B, 1, V)
        out = SimpleNamespace(logits=torch.cat([z, lg], 1))
        if output_hidden_states:
            layers = [torch.cat([hs[k] for hs in hss]) for k in range(len(hss[0]))]
            out.hidden_states = tuple(torch.cat([h.new_zeros(B, 1, h.size(-1)), h], 1) for h in layers)
        return out


BDH100M = ("d3vmeh/bdh-100m-bytes", "bdh-100m-bytes.pt",
           "40b4d6df420111c16a5ffc681d91f0c02877e029a65f06bcbe6032e4f39ee3e1")


def load_bdh(path: str):
    """Pathway's reference BDH (brain/third_party/bdh.py) from a checkpoint: a file, or ``hf`` for the free 100M
    byte-level BDH on Hugging Face (MIT; downloaded and SHA-256 checked)."""
    import hashlib

    from .third_party import bdh
    if path == "hf":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(BDH100M[0], BDH100M[1])
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        assert h.hexdigest() == BDH100M[2], "the 100M BDH checkpoint is not the one checked"
    ck = torch.load(path, weights_only=True, map_location="cpu")
    sd = ck.get("model", ck)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    D = sd["embed.weight"].size(1)
    nh = sd["encoder"].size(0)
    cfg = bdh.BDHConfig(n_layer=6, n_embd=D, n_head=nh, mlp_internal_dim_multiplier=sd["encoder"].size(2) * nh // D,
                        vocab_size=sd["embed.weight"].size(0), dropout=0.0)
    if isinstance(ck.get("config"), dict) and "n_layer" in ck["config"]:
        cfg.n_layer = int(ck["config"]["n_layer"])
    m = bdh.BDH(cfg)
    m.load_state_dict(sd, strict=True)
    return m.eval()


class BDHLM(torch.nn.Module):
    """Pathway's BDH, frozen, with the interface v5 uses (as BrainLM): logits and the state after each layer."""

    def __init__(self, m):
        super().__init__()
        from types import SimpleNamespace
        self.m = m.requires_grad_(False)
        self.config = SimpleNamespace(hidden_size=m.config.n_embd, num_hidden_layers=m.config.n_layer)

    def get_output_embeddings(self):
        from types import SimpleNamespace
        return SimpleNamespace(weight=self.m.lm_head.T)

    @torch.no_grad()
    def _run(self, idx: torch.Tensor):
        m, C = self.m, self.m.config
        B, T = idx.shape
        D, nh = C.n_embd, C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = m.ln(m.embed(idx).unsqueeze(1))
        hs = [x.view(B, T, D)]
        for _ in range(C.n_layer):                           # BDH.forward, keeping each layer's state
            xs = F.relu(x @ m.encoder)
            y = m.ln(m.attn(Q=xs, K=xs, V=x))
            ys = F.relu(y @ m.encoder_v)
            x = m.ln(x + m.ln((xs * ys).transpose(1, 2).reshape(B, 1, T, N * nh) @ m.decoder))
            hs.append(x.view(B, T, D))
        return x.view(B, T, D) @ m.lm_head, hs

    forward = BrainLM.forward


def _is_letter(b: int) -> bool:
    return b >= 128 or 97 <= (b | 32) <= 122


def split_words(raw: bytes) -> list[tuple[int, int]]:
    """(start, end) byte spans of words, as ``models._words``: runs of letters; any other byte except a space is
    a word of its own."""
    out, i, n = [], 0, len(raw)
    while i < n:
        b = raw[i]
        if b in _SPACE:
            i += 1
        elif _is_letter(b):
            j = i
            while j < n and _is_letter(raw[j]):
                j += 1
            out.append((i, j))
            i = j
        else:
            out.append((i, i + 1))
            i += 1
    return out


_CLAUSE = re.compile(r"(?:, (?:and |but )?| and | but )(?=[a-z])")
_WORD = re.compile(r"[A-Za-z]+")


def clause_starts(text: str) -> list[int]:
    """Character offsets where a new clause starts: after ", " or " and " before a small letter, and after . ! ?
    or a new line. (", Kipos." stays in its clause.)"""
    out = [m.end() for m in _CLAUSE.finditer(text)] + [i + 1 for i, c in enumerate(text) if ord(c) in _STOP]
    return sorted(out)


def pool_states(h: torch.Tensor, text: str, cstarts: list[int], who: Who, mu: torch.Tensor) -> torch.Tensor:
    """v7's meaning (``Reader.pool``): next to each step's own state, the mean of its clause's states so far,
    leaving out the words that are names. "Zubak has a dog named" and "Zubak's dog is called" then share "dog", not
    only their last word; the own state still tells the steps inside a name apart. ``h``: [L, d], the states of the
    tokens starting at ``cstarts``; returns [L, 2d], both halves less ``mu`` and of unit length."""
    bounds = clause_starts(text)
    names = []
    for m in _WORD.finditer(text):
        w = m.group()
        if w[0].isupper() and who.word(w.encode())[1] ** who.power >= who.gate:
            names.append((m.start(), m.end()))
    x = h - mu
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    seg = torch.tensor([bisect.bisect_right(bounds, c) for c in cstarts])
    keep = torch.tensor([not any(a <= c < b for a, b in names) for c in cstarts], dtype=x.dtype)
    A = (seg[:, None] == seg[None, :]).to(x.dtype).tril() * keep[None, :]
    n = A.sum(1, keepdim=True)
    eye = torch.eye(len(cstarts), dtype=x.dtype)
    A = torch.where(n > 0, A / n.clamp(min=1), eye)                 # nothing to pool: the step's own state
    m = A @ x
    return torch.cat([m / (m.norm(dim=-1, keepdim=True) + 1e-6), x], -1)


class Who:
    """The who part of the address: the sentence's words so far (the last one perhaps in part), each weighted by
    novelty ** power, as a sparse code. ``base`` is a Brain address (``hybrid.make_address``)."""

    def __init__(self, base: Brain, power: float = 16.0, active: int = 40, gate: float = 0.5, first: bool = False):
        self.base, self.power, self.active, self.gate = base, power, active, gate
        self.first = first                  # v6 option: only the words up to the sentence's first new name count
        self.U = base.cfg.key_size
        self.mult = [pow(257, i, NOVELTY_BUCKETS) for i in range(MAX_WORD)]
        self._total = float(base.key_count.sum())
        self._logN = math.log(1 + self._total)
        self._cache: dict[bytes, tuple[torch.Tensor, float]] = {}

    def word(self, raw: bytes) -> tuple[torch.Tensor, float]:
        """A word's code (as Brain.word_code's C) and how new it is (as Brain.novelty)."""
        if raw not in self._cache:
            pos = [min(i, MAX_WORD - 1) for i in range(len(raw))]
            code = (self.base.key_byte[list(raw)] * self.base.key_inword[pos]).sum(0) / math.sqrt(len(raw))
            h = sum((b * self.mult[p]) % NOVELTY_BUCKETS for b, p in zip(raw, pos)) % NOVELTY_BUCKETS
            nov = math.log((1 + self._total) / (1 + float(self.base.key_count[h]))) / self._logN
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[raw] = (code, nov)
        return self._cache[raw]

    def keys(self, text: str, ends: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """At each byte position in ``ends`` (−1: none): (unit indices [P, active], values [P, active], gate [P])."""
        raw = text.encode()
        spans = split_words(raw)
        P = len(ends)
        kidx = torch.zeros(P, self.active, dtype=torch.long)
        kval = torch.zeros(P, self.active)
        gate = torch.zeros(P, dtype=torch.bool)
        for r, b in enumerate(ends):
            if b < 0:
                continue
            acc, top = None, 0.0
            for s, e in spans:
                if s > b:
                    break
                w = raw[s:min(e, b + 1)]
                if len(w) == 1 and w[0] in _STOP:
                    acc, top = None, 0.0                    # a new sentence starts after . ! ? or a new line
                    continue
                if self.first and top >= self.gate:
                    continue
                code, nov = self.word(w)
                wt = nov ** self.power
                if wt < 1e-3:
                    continue
                acc = wt * code if acc is None else acc + wt * code
                top = max(top, wt)
            if acc is None or top < self.gate:
                continue
            t = acc.topk(self.active)
            v = F.relu(t.values)
            kidx[r], kval[r], gate[r] = t.indices, v / (v.norm() + 1e-6), True
        return kidx, kval, gate


class Reader(torch.nn.Module):
    """What childhood learns: the meaning projection W, the write rate and the read gain. Plus fixed parts: the
    hidden-state mean and the value codes Z (unit-length token embeddings through a fixed random projection)."""

    def __init__(self, d: int, r: int = 32, m: int = 128, layer: int = 8, seed: int = 7, pool: bool = False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layer, self.r, self.m = layer, r, m
        self.pool = pool                                    # v7: W reads pool_states' [clause mean, own state]
        din = 2 * d if pool else d
        self.W = torch.nn.Parameter(torch.randn(din, r, generator=g) / math.sqrt(din))
        self.rate_logit = torch.nn.Parameter(torch.tensor(0.0))           # rate = sigmoid: 0.5 at birth
        self.log_gain = torch.nn.Parameter(torch.tensor(math.log(8.0)))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("P", torch.randn(d, m, generator=g) / math.sqrt(m))

    def what(self, h: torch.Tensor) -> torch.Tensor:
        x = h if self.pool else h - self.mu                  # pooled states come centred
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        w = x @ self.W
        return w / (w.norm(dim=-1, keepdim=True) + 1e-6)

    def Z(self, E: torch.Tensor) -> torch.Tensor:
        z = E @ self.P
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)

    @property
    def rate(self):
        return torch.sigmoid(self.rate_logit)

    @property
    def gain(self):
        return self.log_gain.exp()


# -- reading texts the v5 way ----------------------------------------------------------------------------------------
class Parsed:
    """A batch of texts: the full-context tokens, and for each token its who key, gate, sentence, and (filled by
    ``Student.meaning``) its what vector from the sentence read alone."""

    def __init__(self, tok, who: Who, texts: list[str]):
        self.tk = Tokens(tok, texts)
        B, T = self.tk.ids.shape
        self.kidx = torch.zeros(B, T, who.active, dtype=torch.long)
        self.kval = torch.zeros(B, T, who.active)
        self.gate = torch.zeros(B, T, dtype=torch.bool)
        self.sent = torch.full((B, T), -1, dtype=torch.long)
        self.groups: list[tuple[int, list[int]]] = []                    # (row, token positions) of gated sentences
        for i, t in enumerate(texts):
            raw = t.encode()
            ends = self.tk.ends[i].tolist()
            ki, kv, g = who.keys(t, ends)
            self.kidx[i], self.kval[i], self.gate[i] = ki, kv, g
            # sentence of each token: the number of stops before its first byte (one after the last token's end)
            stops = np.cumsum([0] + [1 if b in _STOP else 0 for b in raw])
            n = self.tk.lens[i]
            sid = [-1] + [int(stops[ends[k - 1] + 1]) if k > 1 else 0 for k in range(1, n)]
            self.sent[i, :n] = torch.tensor(sid)
            for s in sorted(set(sid[1:])):
                pos = [k for k in range(1, n) if sid[k] == s]
                if any(g[k] for k in pos):
                    self.groups.append((i, pos))
        self.what = None                                                    # [B, T, r]
        self.iso_logits = None                                              # gated sentences: logits read alone


class Student(LLMLearner):
    """The frozen LLM plus v5's memory. Learns in life only by the local delta rule."""

    def __init__(self, model, tok, who: Who, reader: Reader):
        super().__init__(model, tok)
        self.who, self.reader = who, reader
        E = model.get_output_embeddings().weight.detach().float().cpu()   # the memory lives on the CPU
        self.E = E / E.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            self.Z = reader.Z(self.E)                                       # V, m
        self.M = torch.zeros(who.U, reader.r, reader.m)                     # the lasting memory, fixed size
        self.ms: list[float] = []
        self.loss_ref = 1.0

    # the sentence read alone gives the meaning
    @torch.no_grad()
    def meaning(self, ps: Parsed, want_logits: bool = False) -> None:
        B, T = ps.tk.ids.shape
        ps.what = torch.zeros(B, T, self.reader.r)
        if want_logits:
            ps.iso_logits = {}
        for j in range(0, len(ps.groups), 32):
            chunk = ps.groups[j:j + 32]
            L = max(len(p) for _, p in chunk) + 1
            ids = torch.full((len(chunk), L), BOS)
            mask = torch.zeros(len(chunk), L, dtype=torch.long)
            for c, (i, pos) in enumerate(chunk):
                ids[c, 1:len(pos) + 1] = ps.tk.ids[i, pos]
                mask[c, :len(pos) + 1] = 1
            logits, hid = run_llm(self.model, ids, mask, self.reader.layer)
            if self.reader.pool:
                P = hid.new_zeros(*hid.shape[:2], 2 * hid.size(-1))
                for c, (i, pos) in enumerate(chunk):
                    cs = [ps.tk.char_starts[i][k] for k in pos]
                    P[c, 1:len(pos) + 1] = pool_states(hid[c, 1:len(pos) + 1], ps.tk.texts[i], cs, self.who,
                                                       self.reader.mu)
                hid = P
            w = self.reader.what(hid)
            for c, (i, pos) in enumerate(chunk):
                ps.what[i, pos] = w[c, 1:len(pos) + 1]
                if want_logits:
                    ps.iso_logits[(i, pos[0])] = logits[c, 1:len(pos) + 1]

    def read(self, kidx, kval, what, M=None) -> torch.Tensor:
        """[N, active], [N, active], [N, r] → [N, m]."""
        M = self.M if M is None else M
        out = torch.zeros(kidx.size(0), self.reader.m)
        for s in range(0, kidx.size(0), 256):
            rows = M[kidx[s:s + 256]]                                       # n, active, r, m
            out[s:s + 256] = torch.einsum("na,nr,narm->nm", kval[s:s + 256], what[s:s + 256], rows)
        return out

    def bias_full(self, ps: Parsed, rows: slice) -> torch.Tensor:
        ids = ps.tk.ids[rows]
        B, T = ids.shape
        g = ps.gate[rows]
        bias = torch.zeros(B, T, self.Z.size(0))
        if g.any():
            r = self.read(ps.kidx[rows][g], ps.kval[rows][g], ps.what[rows][g])
            bias[g] = float(self.reader.gain) * (r @ self.Z.T)
        return bias

    @torch.no_grad()
    def loglik(self, prompts: list[str], conts: list[str]) -> torch.Tensor:
        texts = [p + c for p, c in zip(prompts, conts)]
        ps = Parsed(self.tok, self.who, texts)
        self.meaning(ps)
        tk = ps.tk
        scores = torch.zeros(len(texts))
        for i in range(0, len(texts), 32):
            sl = slice(i, i + 32)
            lg = run_llm(self.model, tk.ids[sl], tk.mask[sl]) + self.bias_full(ps, sl)
            lp = F.log_softmax(lg, dim=-1)
            for j in range(lp.size(0)):
                n, start = i + j, len(prompts[i + j])
                first = next(k for k, c in enumerate(tk.char_starts[n]) if k > 0 and c >= start)
                tgt = tk.ids[n, first:tk.lens[n]]
                scores[n] = lp[j, first - 1:tk.lens[n] - 1].gather(1, tgt[:, None]).sum()
        return scores

    @torch.no_grad()
    def bpb(self, stories: list[str]) -> float:
        texts = [data._fit(s, data.SESSION_BYTES) for s in stories]
        ps = Parsed(self.tok, self.who, texts)
        self.meaning(ps)
        tk = ps.tk
        tot = 0.0
        for i in range(0, len(texts), 32):
            sl = slice(i, i + 32)
            lg = run_llm(self.model, tk.ids[sl], tk.mask[sl]) + self.bias_full(ps, sl)
            y, m = tk.ids[sl, 1:], tk.mask[sl, 1:]
            nll = F.cross_entropy(lg[:, :-1].flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            tot += float((nll * m).sum())
        return tot / sum(len(t.encode()) for t in texts) / float(np.log(2))

    # -- the local rule -------------------------------------------------------------------------------------------
    def write_positions(self, ps: Parsed, row: int = 0):
        """Gated steps whose next token is in the same sentence: (positions, next tokens)."""
        n = ps.tk.lens[row]
        pos = [k for k in range(1, n - 1) if ps.gate[row, k] and ps.sent[row, k + 1] == ps.sent[row, k]]
        return pos, ps.tk.ids[row, [k + 1 for k in pos]]

    @torch.no_grad()
    def learn(self, s: data.Session):
        t0 = time.perf_counter()
        ps = Parsed(self.tok, self.who, [s.text])
        if ps.groups:
            self.meaning(ps, want_logits=True)
            pos, nxt = self.write_positions(ps)
            if pos:
                # surprise: the LLM's loss on the next token, the sentence read alone, with the memory
                where = {k: ((r, pp[0]), j) for r, pp in ps.groups for j, k in enumerate(pp)}
                nll = torch.zeros(len(pos))
                r = self.read(ps.kidx[0, pos], ps.kval[0, pos], ps.what[0, pos])
                b = float(self.reader.gain) * (r @ self.Z.T)
                for q, k in enumerate(pos):
                    key, j = where[k]
                    nll[q] = F.cross_entropy((ps.iso_logits[key][j] + b[q])[None], nxt[q][None])
                surprise = (nll / self.loss_ref).clamp(0, CAP)
                beta = (float(self.reader.rate) * surprise * s.reward).clamp(max=1.0)
                self.M = write(self.M, ps.kidx[0, pos], ps.kval[0, pos], ps.what[0, pos], self.Z[nxt], beta, self)
        self.ms.append((time.perf_counter() - t0) * 1000)

    def memory_bytes(self) -> int:
        return self.M.numel() * 4


def gram(kidx, kval, what, kidx2=None, kval2=None, what2=None) -> torch.Tensor:
    """(k_t · k_s) × (w_t · w_s): how alike two full addresses are."""
    kidx2 = kidx if kidx2 is None else kidx2
    kval2 = kval if kval2 is None else kval2
    what2 = what if what2 is None else what2
    U = int(max(kidx.max(), kidx2.max())) + 1
    a = torch.zeros(kidx.size(0), U).scatter_(-1, kidx, kval)
    b = torch.zeros(kidx2.size(0), U).scatter_(-1, kidx2, kval2)
    return (a @ b.T) * (what @ what2.T)


def solve(kidx, kval, what, V, beta, err0) -> torch.Tensor:
    """The exact sequential delta rule over steps: E = (I + tril(G diag(β), -1))⁻¹ err0."""
    A = torch.tril(gram(kidx, kval, what) * beta[None, :], diagonal=-1)
    return torch.linalg.solve_triangular(A, err0, upper=False, unitriangular=True)


def write(M, kidx, kval, what, V, beta, student: Student) -> torch.Tensor:
    err0 = V - student.read(kidx, kval, what, M)
    E = solve(kidx, kval, what, V, beta, err0)
    upd = torch.einsum("na,nr,nm->narm", kval, what, beta[:, None] * E)   # n, active, r, m
    U, r, m = M.shape
    return M.view(U, -1).index_add(0, kidx.reshape(-1), upd.reshape(-1, r * m)).view(U, r, m)


# -- childhood: learn W, the rate and the gain (the LLM frozen; train names, train relations) -----------------------
def episode(rng: random.Random, names: list[str], relations: list[str], people: int = 8, per: int = 5):
    """A made-up world: (lesson sentences, questions [(prompt, answer)]). Each fact is taught once, in one wording,
    and asked once: in another wording (2 of 3) or the same one (1 of 3)."""
    who = rng.sample(names, people + people * per)
    lessons, questions = [], []
    for p in range(people):
        s = who[p]
        for j, rel in enumerate(rng.sample(relations, per)):
            o = who[people + p * per + j]
            ws = RELATIONS[rel]
            a = rng.randrange(len(ws))
            b = a if rng.random() < 1 / 3 else rng.choice([x for x in range(len(ws)) if x != a])
            lessons.append(ws[a].format(s=s, o=o))
            q = ws[b].format(s=s, o=o)
            questions.append((q[: q.rindex(" " + o)], o))
    rng.shuffle(lessons)
    return lessons, questions


def childhood(args, make_episode=None, dev_episodes=None) -> None:
    """make_episode(rng, names) and dev_episodes(names) (v6) replace v5's episodes when given."""
    torch.manual_seed(0)
    torch.set_num_threads(4)
    sp = data.splits(args.data_dir, need_train=False)
    tok, llm = llm_on(args.llm, args.device)
    who = Who(make_address(0.0, sp.train[:20000]), power=args.power, first=getattr(args, "who", "all") == "first")
    reader = Reader(llm.config.hidden_size, r=args.r, layer=args.layer, pool=bool(getattr(args, "pool", False)))
    st = Student(llm, tok, who, reader)
    # the hidden-state mean, from ordinary sentences
    sents = [s for x in sp.train[:300] for s in data._sentences(x)][:800]
    tk = Tokens(tok, sents)
    hs, n = torch.zeros(llm.config.hidden_size), 0
    with torch.no_grad():
        for i in range(0, len(sents), 32):
            _, hid = run_llm(llm, tk.ids[i:i + 32], tk.mask[i:i + 32], args.layer)
            m = tk.mask[i:i + 32].clone()
            m[:, 0] = 0
            hs += (hid * m[..., None]).sum((0, 1))
            n += int(m.sum())
    reader.mu.copy_(hs / n)
    if getattr(args, "init", None):                     # v6: start from an earlier childhood's reader
        reader.load_state_dict(torch.load(Path(args.init) / "reader.pt", weights_only=False)["reader"])
    st.loss_ref = loss_ref(llm, tok, sp.val[:100])
    opt = torch.optim.Adam(reader.parameters(), lr=args.lr)
    rng = random.Random(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    dev_rng = random.Random(99)
    dev_eps = (dev_episodes(sp.names["dev"]) if dev_episodes else
               [episode(dev_rng, sp.names["dev"], DEV_RELATIONS, people=6, per=4) for _ in range(6)])

    def run_episode(lessons, questions, train: bool):
        """Write the lessons into an empty memory (differentiable), then ask; returns (loss, recall at 1 of 5)."""
        ps = Parsed(tok, who, lessons)
        with torch.no_grad():
            st.meaning(ps, want_logits=True)
        # what again with gradients (the hidden states are fixed; only W changes)
        H = {}
        with torch.no_grad():
            for j in range(0, len(ps.groups), 64):
                chunk = ps.groups[j:j + 64]
                L = max(len(p) for _, p in chunk) + 1
                ids = torch.full((len(chunk), L), BOS)
                mask = torch.zeros(len(chunk), L, dtype=torch.long)
                for c, (i, pos) in enumerate(chunk):
                    ids[c, 1:len(pos) + 1] = ps.tk.ids[i, pos]
                    mask[c, :len(pos) + 1] = 1
                _, hid = run_llm(llm, ids, mask, args.layer)
                for c, (i, pos) in enumerate(chunk):
                    H[(i, pos[0])] = hid[c, 1:len(pos) + 1].float()
                    if reader.pool:
                        H[(i, pos[0])] = pool_states(H[(i, pos[0])], ps.tk.texts[i],
                                                     [ps.tk.char_starts[i][k] for k in pos], who, reader.mu)
        K_i, K_v, Wh, V, NLL = [], [], [], [], []
        for (i, pos) in ps.groups:
            key = (i, pos[0])
            lg = ps.iso_logits[key]
            for j, k in enumerate(pos[:-1]):
                if not ps.gate[i, k]:
                    continue
                K_i.append(ps.kidx[i, k])
                K_v.append(ps.kval[i, k])
                Wh.append(H[key][j])
                nxt = ps.tk.ids[i, pos[j + 1]]
                V.append(st.Z[nxt])
                NLL.append(F.cross_entropy(lg[j][None], nxt[None]))
        if not K_i:
            return None, None
        kidx, kval = torch.stack(K_i), torch.stack(K_v)
        what = reader.what(torch.stack(Wh))
        Vt = torch.stack(V)
        beta = (reader.rate * (torch.stack(NLL) / st.loss_ref).clamp(0, CAP)).clamp(max=1.0)
        Err = solve(kidx, kval, what, Vt, beta, Vt)                         # empty memory: err0 = V
        # the questions, with their answers, in a fresh context
        texts = [q + " " + a + "." for q, a in questions]
        qs = Parsed(tok, who, texts)
        Hq = {}
        with torch.no_grad():
            base, Hq = run_llm(llm, qs.tk.ids, qs.tk.mask, args.layer)
            if reader.pool:
                P = Hq.new_zeros(*Hq.shape[:2], 2 * Hq.size(-1))
                for n in range(len(texts)):
                    L = qs.tk.lens[n]
                    P[n, 1:L] = pool_states(Hq[n, 1:L], texts[n], qs.tk.char_starts[n][1:L], who, reader.mu)
                Hq = P
        loss, hits = 0.0, []
        for n, (q, _) in enumerate(questions):
            first = next(k for k, c in enumerate(qs.tk.char_starts[n]) if k > 0 and c >= len(q))
            last = qs.tk.lens[n] - 1                                         # the "." token
            steps = list(range(first - 1, last - 1))                         # predicting the answer's tokens
            g = qs.gate[n, steps]
            lg = base[n, steps].clone()
            if g.any():
                gs = [s for s, ok in zip(steps, g.tolist()) if ok]
                wq = reader.what(Hq[n, gs])
                Gq = gram(qs.kidx[n, gs], qs.kval[n, gs], wq, kidx, kval, what)   # q, n
                r = Gq @ (beta[:, None] * Err)
                lg[g] = lg[g] + reader.gain * (r @ st.Z.T)
            tgt = qs.tk.ids[n, first:last]
            loss = loss + F.cross_entropy(lg, tgt)
            if not train:
                # recall at 1 of 5: the right answer against 4 other answers of this episode, by summed log-prob
                lp = F.log_softmax(lg, -1).gather(1, tgt[:, None]).sum()
                hits.append(float(lp))
        return loss / len(questions), hits

    def dev_check():
        """Mean answer loss on dev relations (dev names), and how often the answer's first token is the top one
        of the question's own answer and 4 others (approximate recall)."""
        with torch.no_grad():
            tot = 0.0
            for les, qs in dev_eps:
                lo, _ = run_episode(les, qs, train=False)
                tot += float(lo)
        return tot / len(dev_eps)

    best, t0 = float("inf"), time.time()
    for step in range(args.steps + 1):
        if step % args.check == 0:
            d = dev_check()
            log.append({"step": step, "dev_loss": d, "rate": float(reader.rate), "gain": float(reader.gain),
                        "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
            if d < best:
                best = d
                torch.save({"reader": reader.state_dict(), "args": vars(args), "loss_ref": st.loss_ref},
                           out / "reader.pt")
            (out / "log.json").write_text(json.dumps(log, indent=1))
        if step == args.steps:
            break
        les, qs = (make_episode(rng, sp.names["train"]) if make_episode else
                   episode(rng, sp.names["train"], TRAIN_RELATIONS))
        loss, _ = run_episode(les, qs, train=True)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()


def load_student(run: str, llm_name: str = LLM, data_dir: str = "brain/out/data", device: str = "cpu"):
    ck = torch.load(Path(run) / "reader.pt", weights_only=False)
    a = ck["args"]
    sp = data.splits(data_dir, need_train=False)
    tok, llm = llm_on(llm_name, device)
    who = Who(make_address(0.0, sp.train[:20000]), power=a["power"], first=a.get("who", "all") == "first")
    reader = Reader(llm.config.hidden_size, r=a["r"], layer=a["layer"], pool=bool(a.get("pool", False)))
    reader.load_state_dict(ck["reader"])
    st = Student(llm, tok, who, reader)
    st.loss_ref = ck["loss_ref"]
    return st, sp, tok, llm


# -- the school exam, for any learner --------------------------------------------------------------------------------
GURU_QUESTIONS = Path(__file__).with_name("guru_questions.json")


def guru_wordings() -> dict[str, list[str]] | None:
    """Question wordings written by an outside LLM (brain/notes/guru_questions.py), for evaluation only."""
    return json.loads(GURU_QUESTIONS.read_text())["wordings"] if GURU_QUESTIONS.exists() else None


def school_exam(learner, world, two_step_think: bool = True) -> dict:
    """plain, other words (the school's exam wordings, never in a lesson or in childhood), two-step asked directly,
    and two-step thinking out loud: first "{s}'s best friend is" (1 of 5), then the chosen friend's fact (1 of 5)."""
    from .school import ASK, TWO_STEP, WORDINGS, choices, known
    facts = list(world.facts.values())
    plain = [(f.fill(ASK[f.relation]), f.obj) for f in facts]
    other = [(f.fill(WORDINGS[f.relation][1][0]), f.obj) for f in facts]
    two = []
    step1 = []
    for s in world.people:
        friend = world.of(s, "friend").obj
        for rel, q in TWO_STEP.items():
            f2 = world.of(friend, rel)
            two.append((q.format(s=s, p=f2.pet), f2.obj))
            step1.append((s, rel, f2))
    res = {"plain": float(np.mean(known(learner, world, plain))),
           "other_words": float(np.mean(known(learner, world, other))),
           "two_step_direct": float(np.mean(known(learner, world, two)))}
    gw = guru_wordings()
    if gw:                                                  # reported only: one outside wording per fact
        guru = [(gw[f.relation][i % len(gw[f.relation])].format(s=f.subject, p=f.pet), f.obj)
                for i, f in enumerate(facts)]
        res["other_words_guru"] = float(np.mean(known(learner, world, guru)))
    if two_step_think:
        # step 1: which of 5 is the friend?
        prompts, conts, rows = [], [], []
        for s, _, _ in step1:
            q = ASK["friend"].format(s=s)
            opts = choices(world, world.of(s, "friend").obj, q)
            rows.append(opts)
            for o in opts:
                prompts.append(q)
                conts.append(f" {o}.")
        sc = learner.loglik(prompts, conts).view(len(step1), 5)
        picked = [rows[i][int(sc[i].argmax())] for i in range(len(step1))]
        items = []
        for (_, rel, f2), fr in zip(step1, picked):
            pet = world.facts[(fr, "pet")].pet if (fr, "pet") in world.facts else f2.pet
            items.append((ASK[rel].format(s=fr, p=pet), f2.obj))
        res["two_step_think"] = float(np.mean(known(learner, world, items)))
    return res


def run_school(learner, world, guru, stories, lessons: int = 1, seed: int = 0) -> dict:
    """As school.school: each fact taught once in a story, an ordinary story after each lesson; the exam; then one
    quiz per fact with a correction for each wrong answer; the exam again."""
    from .school import WORDINGS, known
    rng = random.Random(seed)
    facts = list(world.facts.values())
    order = [(f, k) for f in facts for k in range(lessons)]
    rng.shuffle(order)
    t0 = time.time()
    for f, k in order:
        learner.learn(data.Session(guru.lesson(f, k), [f.fill(WORDINGS[f.relation][0][0])]))
        learner.learn(data.Session(data._fit(rng.choice(stories), data.SESSION_BYTES)))
    before = school_exam(learner, world)
    wrong = [f for f, ok in zip(facts, known(learner, world, [(guru.quiz(f), f.obj) for f in facts])) if not ok]
    for f in wrong:
        learner.learn(data.Session(data.with_facts(rng.choice(stories), [guru.correction(f)], data.SESSION_BYTES),
                                   [guru.correction(f)], data.REWARD))
    after = school_exam(learner, world)
    return {"before_quiz": before, "after_quiz": after, "corrected": len(wrong), "minutes": (time.time() - t0) / 60}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("childhood")
    c.add_argument("--layer", type=int, default=8)
    c.add_argument("--r", type=int, default=32)
    c.add_argument("--power", type=float, default=16.0)
    c.add_argument("--steps", type=int, default=600)
    c.add_argument("--check", type=int, default=50)
    c.add_argument("--lr", type=float, default=3e-3)
    for sp_ in (c,):
        sp_.add_argument("--llm", default=LLM)
        sp_.add_argument("--data-dir", default="brain/out/data")
        sp_.add_argument("--out", required=True)
    s = sub.add_parser("school")
    s.add_argument("--run", help="a v5 childhood folder; or 'rag' / 'lookup' / 'none'")
    s.add_argument("--names", choices=["dev", "test"], default="dev")
    s.add_argument("--people", type=int, default=40)
    s.add_argument("--llm", default=LLM)
    s.add_argument("--data-dir", default="brain/out/data")
    s.add_argument("--out")
    li = sub.add_parser("life")
    li.add_argument("--run", required=True)
    li.add_argument("--stream", choices=["dev", "test"], default="dev")
    li.add_argument("--val-stories", type=int, default=300)
    li.add_argument("--llm", default=LLM)
    li.add_argument("--data-dir", default="brain/out/data")
    li.add_argument("--out")
    for sub_ in (c, s, li):
        sub_.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    if args.cmd == "childhood":
        childhood(args)
        return
    from .school import TemplateGuru, make_world
    if args.run in ("rag", "lookup", "none"):
        sp = data.splits(args.data_dir, need_train=False)
        tok, llm = llm_on(args.llm, args.device)
        learner = {"rag": lambda: RAG(llm, tok), "lookup": lambda: Lookup(llm),
                   "none": lambda: LLMLearner(llm, tok)}[args.run]()
    else:
        learner, sp, tok, llm = load_student(args.run, args.llm, args.data_dir, args.device)
    if args.cmd == "school":
        world = make_world(sp.names[args.names], args.people, seed=0 if args.names == "dev" else 1)
        stories = sp.life_dev if args.names == "dev" else sp.life_test
        res = run_school(learner, world, TemplateGuru(world, stories), stories)
        res.update(run=args.run, names=args.names, people=args.people)
    else:
        from .lifetest import SEEDS
        sizes = data.FULL_LIFE
        names = sp.names[args.stream]
        stories = sp.life_dev if args.stream == "dev" else sp.life_test
        stream = data.life_stream(stories, names, SEEDS[args.stream], sizes)
        stream.heldout = args.stream == "test"
        res = live(learner, stream, sp.val[: args.val_stories])
        gw = guru_wordings()
        if gw:                                              # reported only: one outside wording per main fact
            from .lifetest import recall
            from .school import WORDINGS
            rels = list(WORDINGS)                           # the same order as data.RELATIONS
            for i, f in enumerate(stream.facts["main"]):
                ws = gw[rels[f.relation]]
                object.__setattr__(f, "_guru", ws[i % len(ws)].format(s=f.subject, p=f.pet))
            orig = data.Fact.heldout
            try:
                data.Fact.heldout = property(lambda self: self._guru)
                res["recall_guru"] = float(np.mean(recall(learner, stream, stream.facts["main"], "heldout")))
            finally:
                data.Fact.heldout = orig
        res["main_correct"] = [bool(x) for x in res["main_correct"]]
        if hasattr(learner, "ms"):
            res["ms_per_session_median"] = float(np.median(learner.ms))
        if hasattr(learner, "memory_bytes"):
            res["memory_bytes"] = learner.memory_bytes()
        res.update(run=args.run, stream=args.stream)
    print(json.dumps({k: v for k, v in res.items() if k != "main_correct"}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
