"""Teach Shishya something and ask it later. No training runs: it only changes its memory, by a local rule, as it reads.

    pip install torch numpy huggingface_hub
    python -m brain.teach               # teach it, ask it
    python -m brain.teach --tour        # a fixed tour: what it can do, and what it can't

A line ending in "." is taught. Anything else is a sentence for it to finish, with nothing else in front of it, so
what it knows can only come from its memory. The first run downloads a free 400 MB model (Pathway's 100M BDH, MIT,
SHA-256 checked). Everything runs on a CPU. The model is v7 (docs/V7.md), and it writes one byte at a time, so an
answer is slow: 2-3 s per letter on a busy 4-core machine.

This is not a measurement. The results are in docs/RESEARCH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch

from . import data, v5
from .hybrid import run_llm

WEIGHTS = Path(__file__).with_name("weights")
READER = WEIGHTS / "v7_reader.pt"
ADDRESS = WEIGHTS / "v7_address.pt"
MAX_ANSWER = 16      # bytes
STOP = ".\n!?,"
NOT_STORED = ("Nothing stored. It only stores a sentence with a word that is new to it, such as a rare name, "
              "and every word here looks familiar.")


def load_address():
    """The fixed code behind the memory's address. Only the word counts are saved: the rest comes from a fixed seed."""
    from .hybrid import make_address
    ck = torch.load(ADDRESS, weights_only=True)
    base = make_address(0.0, [])
    base.key_count[ck["index"].long()] = ck["count"].float()
    return base


def load(bdh: str = "hf", threads: int = 0) -> v5.Student:
    """The frozen 100M BDH, v7's reader and an empty memory."""
    if threads:
        torch.set_num_threads(threads)
    ck = torch.load(READER, weights_only=True)
    a = ck["args"]
    model = v5.BDHLM(v5.load_bdh(bdh))
    who = v5.Who(load_address(), power=a["power"], first=a["who"] == "first")
    reader = v5.Reader(model.config.hidden_size, r=a["r"], layer=a["layer"], pool=a["pool"])
    reader.load_state_dict(ck["reader"])
    st = v5.Student(model, v5.ByteTok(), who, reader)
    st.loss_ref = ck["loss_ref"]
    return st


def fingerprint(st: v5.Student) -> str:
    """A hash of every weight of the network, to show that reading doesn't change them."""
    h = hashlib.sha256()
    for p in st.model.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def teach(st: v5.Student, sentence: str) -> dict:
    """Read one sentence into the memory. Returns whether anything was stored, and the time and rows touched."""
    t0 = time.perf_counter()
    before = int((st.M != 0).flatten(1).any(1).sum())
    stored = bool(v5.Parsed(st.tok, st.who, [sentence]).gate.any())
    st.learn(data.Session(sentence))
    return {"stored": stored, "seconds": time.perf_counter() - t0,
            "new_rows": int((st.M != 0).flatten(1).any(1).sum()) - before}


@torch.no_grad()
def answer(st: v5.Student, prompt: str, memory: bool = True, on_byte=None) -> str:
    """Finish ``prompt``, one byte at a time, always taking the likeliest. Stops at the end of a clause."""
    raw = prompt.encode()
    out = b""
    for _ in range(MAX_ANSWER):
        text = (raw + out).decode("utf-8", "ignore")
        if memory:
            ps = v5.Parsed(st.tok, st.who, [text])
            st.meaning(ps)
            ids, mask = ps.tk.ids, ps.tk.mask
            logits = run_llm(st.model, ids, mask) + st.bias_full(ps, slice(0, 1))
        else:
            tk = v5.Tokens(st.tok, [text])
            ids, mask = tk.ids, tk.mask
            logits = run_llm(st.model, ids, mask)
        byte = int(logits[0, int(mask[0].sum()) - 1].argmax())
        out += bytes([byte])
        if on_byte:
            on_byte(bytes([byte]))
        if chr(byte) in STOP:
            break
    return out.decode("utf-8", "replace")


class Show:
    def __init__(self, st: v5.Student):
        self.st = st

    def say(self, s: str = "") -> None:
        print(s, flush=True)

    def teach(self, sentence: str) -> None:
        self.say(f"read  {sentence}")
        r = teach(self.st, sentence)
        self.say(f"      stored in {r['seconds']:.1f} s" if r["stored"] else f"      {NOT_STORED}")

    def ask(self, prompt: str) -> None:
        self.say(f"ask   {prompt} ...")
        self.say(f"      before reading: {answer(self.st, prompt, memory=False).strip()}")
        print("      now:            ", end="", flush=True)
        started = []

        def show(b: bytes) -> None:
            if started or b != b" ":                   # the answer's first space is only the gap after the prompt
                started.append(b)
                sys.stdout.buffer.write(b)
                sys.stdout.buffer.flush()

        answer(self.st, prompt, on_byte=show)
        self.say()


def tour(st: v5.Student) -> None:
    s = Show(st)
    weights = fingerprint(st)
    s.say("Each answer starts fresh: only the unfinished sentence goes in, with nothing before it.\n")
    s.say("# It reads each sentence once. No training runs.")
    for f in ("Priya lives in a town called Nashik.", "Zubak has a dog named Rex.", "Rahul's favourite food is pizza."):
        s.teach(f)
    s.say("\n# It knows them")
    for q in ("Priya lives in a town called", "Zubak has a dog named", "Rahul's favourite food is"):
        s.ask(q)
    s.say("\n# In words it has not read")
    for q in ("The town where Priya lives is called", "The name of Zubak's dog is"):
        s.ask(q)
    s.say("\n# What it gets wrong: a change said in other words doesn't reach the old fact")
    s.teach("Priya moved to a town called Pune.")
    s.ask("Priya lives in a town called")
    s.say("\n# The same change, said the way the fact was first said, works")
    s.teach("Priya lives in a town called Pune.")
    s.ask("Priya lives in a town called")
    s.say("\n# What it gets wrong: a common name looks like nothing new, so nothing is stored")
    s.teach("Marie works at a company called Globex.")
    s.ask("Marie works at a company called")
    same = fingerprint(st) == weights
    used = int((st.M != 0).flatten(1).any(1).sum())
    s.say(f"\nThe network's weights after all this: {'unchanged' if same else 'CHANGED'} (hash {weights}).")
    s.say(f"Only the memory changed: {used:,} of its {st.M.size(0):,} rows.")


def repl(st: v5.Student, memory: Path | None) -> None:
    s = Show(st)
    s.say('Teach it with a line ending in "."  Ask by leaving a sentence unfinished. Ctrl-D quits.')
    s.say("Rare names work best. Example:  Zubak has a dog named Rex.   then:  Zubak has a dog named\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line:
            s.teach(line) if line.endswith(".") else s.ask(line)
    if memory:
        torch.save(st.M.to_sparse(), memory)
        s.say(f"\nMemory saved to {memory}.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tour", action="store_true", help="a fixed tour, then exit")
    p.add_argument("--memory", type=Path, help="keep what it learns in this file, between runs")
    p.add_argument("--bdh", default="hf", help="the BDH checkpoint: a file, or hf to download it (default)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")
    a = p.parse_args(argv)
    print("Loading (the first run downloads a free 400 MB model) ...", flush=True)
    st = load(a.bdh, a.threads)
    if a.memory and a.memory.exists():
        st.M = torch.load(a.memory, weights_only=True).to_dense()
        print(f"Memory loaded from {a.memory}.")
    print()
    tour(st) if a.tour else repl(st, a.memory)


if __name__ == "__main__":
    main()
