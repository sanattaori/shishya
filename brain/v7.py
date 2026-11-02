"""Shishya v7: v5/v6's memory plus reconsolidation and sleep. No gradients, awake or asleep (docs/V7.md).

    python -m brain.v7 changes --run <v6 or v5 run> --mode both --names dev

**Awake** it learns as v5 (the local delta rule), and it also keeps a trace of each write: the address (who units,
what vector), the token it wrote and its strength. Traces are numbers, not text. Steps that write the start of a
new name begin a "fact"; the steps that write the rest of that name belong to it.

**Reconsolidation (awake).** Before a new fact is written, Shishya recalls what its memory holds at the new
address. If it recalls the first token of a different name, and that name is a live fact with a close address
(who × what ≥ ``match``), the old fact is replaced: it is marked dead and erased at once (the delta rule toward
nothing at its addresses).

**Sleep, every ``day`` sessions.** The memory is rebuilt from zero out of the live traces, in ``passes`` shuffled
passes of the same delta rule (interleaved replay). Dead facts are gone for good, and facts written in different
orders disturb each other less. With ``resolve`` > 0, sleep first lets the later of two facts at nearly the same
address win (who × what ≥ ``resolve``, different first tokens): the earlier one dies.

Modes: ``wake`` (reconsolidation only), ``sleep`` (sleep only), ``both``, ``none`` (v5/v6 as is).

**Clause meaning** (``clause`` > 0, any mode). A sentence that says several things ("Zubak lives in Kelm, has a dog
called Rex and likes Kipos best of all") blurs each part's meaning with the others. With ``clause``, each clause is
also read alone (after the sentence's first word, its subject), and a step's "what" is ``clause`` × the clause's
meaning + (1 − ``clause``) × the sentence's. A new clause starts after ", " or " and " when the next word starts with
a small letter (", Kipos." stays one clause).
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data, v5
from .hybrid import BOS, run_llm

_CLAUSE = re.compile(r"(?:, (?:and |but )?| and | but )(?=[a-z])")


def clause_starts(text: str) -> list[int]:
    """Character offsets where a new clause starts (inside a sentence; v5.clause_starts also cuts at . ! ?)."""
    return [m.end() for m in _CLAUSE.finditer(text)]


class Sleeper(v5.Student):
    def __init__(self, base: v5.Student, mode: str = "both", day: int = 100, passes: int = 3, match: float = 0.3,
                 recall_min: float = 0.3, seed: int = 0, clause: float = 0.0, resolve: float = 0.0):
        super().__init__(base.model, base.tok, base.who, base.reader)
        self.loss_ref = base.loss_ref
        self.mode, self.day, self.passes, self.match, self.recall_min = mode, day, passes, match, recall_min
        self.clause, self.resolve = clause, resolve
        self.rng = random.Random(seed)
        self.K_i, self.K_v, self.Wh, self.tokens, self.beta, self.fact = [], [], [], [], [], []
        self.alive: dict[int, bool] = {}
        self.first: dict[int, int] = {}                 # fact -> index of its first trace
        self.sessions = self.replaced = self.sleeps = 0

    @torch.no_grad()
    def meaning(self, ps: v5.Parsed, want_logits: bool = False) -> None:
        super().meaning(ps, want_logits)
        if self.clause <= 0:
            return
        parts = []                                        # (row, positions) of each clause of a gated sentence
        for i, pos in ps.groups:
            cs = clause_starts(ps.tk.texts[i])
            if not cs:
                continue
            cut: dict[int, list[int]] = {}
            for k in pos:
                cut.setdefault(bisect.bisect_right(cs, ps.tk.char_starts[i][k]), []).append(k)
            if len(cut) <= 1:
                continue
            text, st = ps.tk.texts[i], ps.tk.char_starts[i]
            c0 = st[pos[0]]
            while c0 < len(text) and text[c0] == " ":
                c0 += 1
            sp = text.find(" ", c0)
            subj = [k for k in pos if st[k] < (len(text) if sp < 0 else sp)]
            for n, ks in enumerate(cut.values()):          # later clauses are read after the sentence's first word
                parts.append((i, ks, subj if n > 0 else []))
        for j in range(0, len(parts), 32):
            chunk = parts[j:j + 32]
            L = max(len(p) + len(sj) for _, p, sj in chunk) + 1
            ids = torch.full((len(chunk), L), BOS)
            mask = torch.zeros(len(chunk), L, dtype=torch.long)
            for c, (i, pos, sj) in enumerate(chunk):
                ids[c, 1:len(sj) + len(pos) + 1] = ps.tk.ids[i, sj + pos]
                mask[c, :len(sj) + len(pos) + 1] = 1
            _, hid = run_llm(self.model, ids, mask, self.reader.layer)
            w = self.reader.what(hid)
            for c, (i, pos, sj) in enumerate(chunk):
                a = 1 + len(sj)
                mix = self.clause * w[c, a:a + len(pos)] + (1 - self.clause) * ps.what[i, pos]
                ps.what[i, pos] = mix / (mix.norm(dim=-1, keepdim=True) + 1e-6)

    # -- which written tokens start or continue a new name ----------------------------------------------------------
    def _name_steps(self, text: str, ps: v5.Parsed, pos: list[int]) -> list[tuple[bool, bool]]:
        """For each written step: (starts a new name, continues the same name)."""
        raw = text.encode()
        spans = v5.split_words(raw)
        tk = ps.tk
        out = []
        for k in pos:
            c = tk.char_starts[0][k + 1]                 # first character of the token written at step k
            b = len(text[:c].encode())
            while b < len(raw) and raw[b] == ord(" "):
                b += 1
            w = next(((s, e) for s, e in spans if s <= b < e), None)
            if w is None:
                out.append((False, False))
                continue
            _, nov = self.who.word(raw[w[0]:w[1]])
            name = raw[w[0]:w[0] + 1].isupper() and nov ** self.who.power >= self.who.gate
            out.append((name and b == w[0], name and b > w[0]))
        return out

    @torch.no_grad()
    def learn(self, s: data.Session):
        t0 = time.perf_counter()
        ps = v5.Parsed(self.tok, self.who, [s.text])
        if ps.groups:
            self.meaning(ps, want_logits=True)
            pos, nxt = self.write_positions(ps)
            if pos:
                where = {k: ((r, pp[0]), j) for r, pp in ps.groups for j, k in enumerate(pp)}
                kidx, kval, what = ps.kidx[0, pos], ps.kval[0, pos], ps.what[0, pos]
                r = self.read(kidx, kval, what)
                b = float(self.reader.gain) * (r @ self.Z.T)
                nll = torch.zeros(len(pos))
                for q, k in enumerate(pos):
                    key, j = where[k]
                    nll[q] = F.cross_entropy((ps.iso_logits[key][j] + b[q])[None], nxt[q][None])
                beta = (float(self.reader.rate) * (nll / self.loss_ref).clamp(0, v5.CAP) * s.reward).clamp(max=1.0)
                names = self._name_steps(s.text, ps, pos)
                if self.mode in ("wake", "both"):
                    self._reconsolidate(kidx, kval, what, r, nxt, names)
                self.M = v5.write(self.M, kidx, kval, what, self.Z[nxt], beta, self)
                self._log(kidx, kval, what, nxt, beta, names)
        self.sessions += 1
        if self.mode in ("sleep", "both") and self.sessions % self.day == 0:
            self.sleep()
        self.ms.append((time.perf_counter() - t0) * 1000)

    def _log(self, kidx, kval, what, nxt, beta, names):
        fact = -1
        for q in range(len(nxt)):
            start, cont = names[q]
            if start:
                fact = len(self.alive)
                self.alive[fact] = True
                self.first[fact] = len(self.tokens)
            elif not cont:
                fact = -1
            self.K_i.append(kidx[q].to(torch.int16))
            self.K_v.append(kval[q].half())
            self.Wh.append(what[q].half())
            self.tokens.append(int(nxt[q]))
            self.beta.append(float(beta[q]))
            self.fact.append(fact)

    def _reconsolidate(self, kidx, kval, what, r, nxt, names):
        """A new name whose address recalls another live name's first token, at a close address: replace it."""
        if not self.tokens:
            return
        live_first = [(f, i) for f, i in self.first.items() if self.alive[f]]
        if not live_first:
            return
        idx = torch.tensor([i for _, i in live_first])
        Ki = torch.stack([self.K_i[i] for i in idx.tolist()]).long()
        Kv = torch.stack([self.K_v[i] for i in idx.tolist()]).float()
        Wo = torch.stack([self.Wh[i] for i in idx.tolist()]).float()
        tok = torch.tensor([self.tokens[i] for i in idx.tolist()])
        for q in range(len(nxt)):
            if not names[q][0]:
                continue
            rn = r[q].norm()
            if rn < 1e-6:
                continue
            sc = (r[q] @ self.Z.T) / rn
            o = int(sc.argmax())
            if o == int(nxt[q]) or float(sc[o]) < self.recall_min:
                continue
            cand = (tok == o).nonzero().flatten()
            if len(cand) == 0:
                continue
            g = v5.gram(kidx[q:q + 1], kval[q:q + 1], what[q:q + 1], Ki[cand], Kv[cand], Wo[cand])[0]
            best = int(g.argmax())
            if float(g[best]) < self.match:
                continue
            f = live_first[int(cand[best])][0]
            self._erase(f)

    def _erase(self, f: int):
        self.alive[f] = False
        self.replaced += 1
        steps = [i for i, x in enumerate(self.fact) if x == f]
        if steps:
            Ki = torch.stack([self.K_i[i] for i in steps]).long()
            Kv = torch.stack([self.K_v[i] for i in steps]).float()
            Wo = torch.stack([self.Wh[i] for i in steps]).float()
            zero = torch.zeros(len(steps), self.reader.m)
            self.M = v5.write(self.M, Ki, Kv, Wo, zero, torch.ones(len(steps)), self)

    def _later_wins(self):
        """At sleep: of two live facts whose first steps have nearly the same address (who × what ≥ ``resolve``)
        and different first tokens, the earlier one dies."""
        facts = [(f, i) for f, i in sorted(self.first.items(), key=lambda t: t[1]) if self.alive[f]]
        if len(facts) < 2:
            return
        idx = [i for _, i in facts]
        Ki = torch.stack([self.K_i[i] for i in idx]).long()
        Kv = torch.stack([self.K_v[i] for i in idx]).float()
        Wo = torch.stack([self.Wh[i] for i in idx]).float()
        tok = torch.tensor([self.tokens[i] for i in idx])
        n = len(facts)
        for a in range(0, n, 512):
            g = v5.gram(Ki[a:a + 512], Kv[a:a + 512], Wo[a:a + 512], Ki, Kv, Wo)        # rows: earlier candidates
            later = torch.arange(n)[None, :] > torch.arange(a, min(a + 512, n))[:, None]
            differ = tok[None, :] != tok[a:a + 512, None]
            dead = ((g >= self.resolve) & later & differ).any(1)
            for q in dead.nonzero().flatten().tolist():
                self.alive[facts[a + q][0]] = False
                self.replaced += 1

    def sleep(self):
        """Rebuild the memory from the live traces: ``passes`` shuffled passes of the delta rule."""
        if self.resolve > 0:
            self._later_wins()
        live = [i for i, f in enumerate(self.fact) if f < 0 or self.alive[f]]
        if not live:
            return
        Ki = torch.stack(self.K_i).long()
        Kv = torch.stack(self.K_v).float()
        Wo = torch.stack(self.Wh).float()
        V = self.Z[torch.tensor(self.tokens)]
        B = torch.tensor(self.beta)
        M = torch.zeros_like(self.M)
        for _ in range(self.passes):
            order = live[:]
            self.rng.shuffle(order)
            for c in range(0, len(order), 256):
                ix = torch.tensor(order[c:c + 256])
                M = v5.write(M, Ki[ix], Kv[ix], Wo[ix], V[ix], B[ix], self)
        self.M = M
        self.sleeps += 1

    def trace_bytes(self) -> int:
        """Stored as int16 units, fp16 values and meaning, a 4-byte token, an fp16 strength, a 4-byte fact id."""
        per = self.who.active * 4 + self.reader.r * 2 + 4 + 2 + 4
        return per * len(self.tokens)

    def end_of_day(self):
        if self.mode in ("sleep", "both"):
            self.sleep()

    def memory_state(self) -> dict:
        return {"M": self.M, "K_i": self.K_i, "K_v": self.K_v, "Wh": self.Wh, "tokens": self.tokens,
                "beta": self.beta, "fact": self.fact, "alive": self.alive, "first": self.first,
                "counts": (self.sessions, self.replaced, self.sleeps)}

    def load_memory_state(self, x: dict):
        self.M, self.K_i, self.K_v, self.Wh = x["M"], x["K_i"], x["K_v"], x["Wh"]
        self.tokens, self.beta, self.fact, self.alive, self.first = (x["tokens"], x["beta"], x["fact"], x["alive"],
                                                                     x["first"])
        self.sessions, self.replaced, self.sleeps = x["counts"]

    def stats(self) -> dict:
        rows = int((self.M.flatten(1).abs().sum(1) > 0).sum())
        return {"mode": self.mode, "replaced": self.replaced, "sleeps": self.sleeps, "traces": len(self.tokens),
                "trace_bytes": self.trace_bytes(), "memory_rows_used": rows,
                "memory_bytes_used_fp16": rows * self.reader.r * self.reader.m * 2}


def make(run: str, llm: str, data_dir: str, device: str, mode: str, **kw):
    st, sp, tok, model = v5.load_student(run, llm, data_dir, device)
    return Sleeper(st, mode, **kw), sp


def main(argv=None):
    from .hardschool import changes, dense
    from .school import TemplateGuru, make_world
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", choices=["changes", "dense", "school", "life", "longlife"])
    p.add_argument("--run", required=True)
    p.add_argument("--mode", choices=["none", "wake", "sleep", "both"], default="both")
    p.add_argument("--names", choices=["dev", "test"], default="dev")
    p.add_argument("--day", type=int, default=100)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--match", type=float, default=0.3)
    p.add_argument("--recall-min", type=float, default=0.3)
    p.add_argument("--clause", type=float, default=0.0)
    p.add_argument("--resolve", type=float, default=0.0)
    p.add_argument("--llm", default=v5.LLM)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    kw = dict(day=args.day, passes=args.passes, match=args.match, recall_min=args.recall_min, clause=args.clause,
              resolve=args.resolve)
    st, sp = make(args.run, args.llm, args.data_dir, args.device, args.mode, **kw)
    seed = 0 if args.names == "dev" else 1
    stories = sp.life_dev if args.names == "dev" else sp.life_test
    if args.test in ("changes", "dense", "school"):
        w = make_world(sp.names[args.names], 40, seed=seed)
        if args.test == "changes":
            res = changes(st, w, stories, sp.names[args.names], seed)
        elif args.test == "dense":
            res = dense(st, w, stories, seed)
        else:
            res = v5.run_school(st, w, TemplateGuru(w, stories), stories)
    elif args.test == "life":
        from .lifetest import SEEDS, live
        stream = data.life_stream(stories, sp.names[args.names], SEEDS[args.names], data.FULL_LIFE)
        stream.heldout = args.names == "test"
        res = live(st, stream, sp.val[:300])
        res["main_correct"] = [bool(x) for x in res["main_correct"]]
    else:
        from . import longlife, v6
        life = longlife.Life(sp.names["test"], sp.life_test, v6.teacher(), v5.guru_wordings())
        out = Path(args.out or "brain/out/v7/longlife.json")
        holder = {}

        def mk():                                   # a new learner each day; memory and traces come from disk
            s = Sleeper(st, args.mode, **{**kw, "day": 10**9})
            holder["s"] = s
            return s
        res = longlife.run(mk, life, sp.val[:300], out.parent / (out.stem + "_mem"))
        st = holder["s"]
    res.update(st.stats(), test=args.test, run=args.run, names=args.names, llm=args.llm, **kw)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("correct") and k != "days"}, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()



SLEEP_SETTINGS = {                                   # v7's dev choice (docs/V7.md); the first wins a tie
    "none": ["--mode", "none"],
    "sleep": ["--mode", "sleep"],
    "sleep_r0.8": ["--mode", "sleep", "--resolve", "0.8"],
    "sleep_r0.9": ["--mode", "sleep", "--resolve", "0.9"],
}


def choose_sleep(rows: dict[str, dict]) -> str:
    """The setting with the best mean of current value (plain + other words) and never-changed facts on hard school
    A with dev names; a tie goes to the earlier one in SLEEP_SETTINGS."""
    def score(k):
        return (rows[k]["current_both"] + rows[k]["kept_plain"]) / 2
    order = list(SLEEP_SETTINGS)
    return max([k for k in order if k in rows], key=lambda k: (round(score(k), 6), -order.index(k)))
