"""Shishya hybrid: a frozen open LLM (the cortex) with Shishya's lasting memory (the hippocampus).

    python -m brain.hybrid --stage dev  --out brain/out/hybrid/dev.json
    python -m brain.hybrid --stage test --dev-report brain/out/hybrid/dev.json --out brain/out/hybrid/test.json

The plan and its pass rules are in docs/HYBRID.md. They were fixed before the dev runs.

- **The LLM** (SmolLM2-135M, Apache-2.0) is never changed: no gradients, no fine-tuning, nothing in its weights.
- **The memory** is v3/v4's: the same fixed sparse address (the words of the sentence, each counting by how new it
  is; v4's meaning part optional) and the same local delta rule, β = rate × surprise × reward. One matrix of
  16,384 units × the LLM's width, fixed in size.
- **What it stores:** at each token, the address of the text so far is linked to the *next token* (its unit-length
  embedding). **How it is read:** the address of the text so far reads back a vector; its likeness to each token's
  embedding, times ``gain``, is added to the LLM's next-token scores. No part of it is trained. ``rate`` and
  ``gain`` are chosen on the dev stream.
- **Surprise** is the LLM's own loss on the token, divided by its average loss on held-out stories, capped at 4.

Baselines, all on the same LLM: no learning, RAG (keep every sentence, put the best match before the question),
fine-tuning (one Adam step per session) and a plain database (lookup, the ceiling for this test).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data
from .lifetest import SEEDS, Learner, Lookup, Retrieval, bootstrap_lower, choose, live, peak_rss_mb
from .models import Brain, BrainConfig, _sparse_read

LLM = "HuggingFaceTB/SmolLM2-135M"
BOS = 0                                        # SmolLM2's <|endoftext|>, put before every text
CAP = 4.0                                      # surprise cap, as the Brain's max_surprise
GRID = {"key_bag": [0.0, 4.0], "rate": [0.25, 0.5, 1.0], "gain": [2.0, 4.0, 8.0]}
LR_GRID = [3e-6, 1e-5, 3e-5]
CHUNK = 32                                     # sequences per forward pass when scoring


def load_llm(name: str = LLM):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def make_address(key_bag: float, familiar: list[str]) -> Brain:
    """Only the Brain's fixed address code (``Brain.window_keys``), with v3's settings and v4's meaning part.
    The word counts come from the first 20,000 childhood stories, as for v3. The tiny network around it is unused."""
    cfg = BrainConfig(n_layer=1, n_embd=8, n_head=1, mult=1, key="words", key_window=8, key_decay=1.0,
                      key_sentence=True, key_novelty=True, key_bag=key_bag)
    a = Brain(cfg).eval()
    a.learn_familiarity(familiar)
    return a


# -- the LLM as a reader ---------------------------------------------------------------------------------------------
class Tokens:
    """Texts tokenized with BOS first, plus each token's last byte (to find its address), padded into a batch."""

    def __init__(self, tok, texts: list[str]):
        ids, ends, self.char_starts = [], [], []
        for t in texts:
            e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
            nb = np.cumsum([0] + [len(ch.encode()) for ch in t])            # char offset -> byte offset
            ids.append([BOS] + e["input_ids"])
            ends.append([-1] + [int(nb[b]) - 1 for _, b in e["offset_mapping"]])
            self.char_starts.append([0] + [a for a, _ in e["offset_mapping"]])
        L = max(map(len, ids))
        self.ids = torch.tensor([x + [BOS] * (L - len(x)) for x in ids])
        self.mask = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
        self.ends = torch.tensor([x + [-1] * (L - len(x)) for x in ends])
        self.lens = [len(x) for x in ids]
        self.texts = texts


class LLMLearner(Learner):
    """A frozen LLM, read with an optional memory bias (``Hybrid``). Scores are in nats, as in lifetest."""

    def __init__(self, model, tok=None):
        super().__init__(model)
        self.tok = tok

    def bias(self, tk: Tokens, rows: slice) -> torch.Tensor | None:
        return None

    def logits(self, tk: Tokens, rows: slice) -> torch.Tensor:
        out = self.model(input_ids=tk.ids[rows], attention_mask=tk.mask[rows]).logits.float()
        b = self.bias(tk, rows)
        return out if b is None else out + b

    @torch.no_grad()
    def loglik(self, prompts: list[str], conts: list[str]) -> torch.Tensor:
        texts = [p + c for p, c in zip(prompts, conts)]
        tk = Tokens(self.tok, texts)
        scores = torch.zeros(len(texts))
        for i in range(0, len(texts), CHUNK):
            lp = F.log_softmax(self.logits(tk, slice(i, i + CHUNK)), dim=-1)
            for j in range(lp.size(0)):
                n, start = i + j, len(prompts[i + j])
                # the continuation's tokens: those starting at or after the end of the prompt
                first = next(k for k, c in enumerate(tk.char_starts[n]) if k > 0 and c >= start)
                tgt = tk.ids[n, first:tk.lens[n]]
                scores[n] = lp[j, first - 1:tk.lens[n] - 1].gather(1, tgt[:, None]).sum()
        return scores

    @torch.no_grad()
    def bpb(self, stories: list[str]) -> float:
        """Bits per byte of the first 256 bytes of each story (every token scored, after BOS)."""
        texts = [data._fit(s, data.SESSION_BYTES) for s in stories]
        tk = Tokens(self.tok, texts)
        tot = 0.0
        for i in range(0, len(texts), CHUNK):
            lg = self.logits(tk, slice(i, i + CHUNK))
            y, m = tk.ids[i:i + CHUNK, 1:], tk.mask[i:i + CHUNK, 1:]
            nll = F.cross_entropy(lg[:, :-1].flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            tot += float((nll * m).sum())
        return tot / sum(len(t.encode()) for t in texts) / float(np.log(2))


class Hybrid(LLMLearner):
    """The frozen LLM plus Shishya's lasting memory, written by the local delta rule only."""

    def __init__(self, model, tok, address: Brain, rate: float, gain: float, loss_ref: float):
        super().__init__(model, tok)
        emb = model.get_output_embeddings().weight.detach().float()
        self.E = emb / emb.norm(dim=-1, keepdim=True)                       # V, d: unit-length token codes
        self.address, self.rate, self.gain, self.loss_ref = address, rate, gain, loss_ref
        self.M = torch.zeros(1, 1, address.cfg.key_size, self.E.size(1))   # the lasting memory, fixed size
        self.ms: list[float] = []

    def keys(self, tk: Tokens, rows: slice):
        """The address at each token's last byte: (indices, values) [B, T, active]; zero at BOS and padding."""
        texts = tk.texts[rows]
        raw = [t.encode() for t in texts]
        Lb = max(1, max(map(len, raw)))
        idx = torch.zeros(len(raw), Lb, dtype=torch.long)
        for i, b in enumerate(raw):
            idx[i, : len(b)] = torch.tensor(list(b), dtype=torch.long)
        kidx, kval = self.address.window_keys(idx)                         # B, Lb, active
        ends = tk.ends[rows]
        g = ends.clamp_min(0)[..., None].expand(-1, -1, kidx.size(-1))
        ki, kv = kidx.gather(1, g), kval.gather(1, g)
        return ki, kv * (ends >= 0)[..., None]

    def bias(self, tk: Tokens, rows: slice) -> torch.Tensor:
        ki, kv = self.keys(tk, rows)
        r = _sparse_read(self.M, ki, kv)[:, 0]                              # B, T, d
        return self.gain * (r @ self.E.T)

    @torch.no_grad()
    def learn(self, s: data.Session):
        t0 = time.perf_counter()
        tk = Tokens(self.tok, [s.text])
        rows = slice(0, 1)
        lg = self.logits(tk, rows)[0]                                       # T, V (with the memory, as the Brain)
        ids = tk.ids[0]
        nll = F.cross_entropy(lg[:-1], ids[1:], reduction="none")          # predicting token t+1 at t
        surprise = (nll / self.loss_ref).clamp(0, CAP)
        ki, kv = self.keys(tk, rows)
        ki, kv = ki[0, :-1], kv[0, :-1]                                     # T-1: the address at t ...
        V = self.E[ids[1:]]                                                 # ... is linked to token t+1
        beta = (self.rate * surprise * s.reward).clamp(max=1.0) * (kv.sum(-1) > 0)
        # the exact sequential delta rule (as Brain._write_window): E = (I + tril(diag(β) K Kᵀ, -1))⁻¹ (V − K M₀)
        U = self.M.size(2)
        dense = torch.zeros(ki.size(0), U).scatter_(-1, ki, kv)
        A = torch.tril((dense @ dense.T) * beta[None, :], diagonal=-1)
        err0 = V - _sparse_read(self.M, ki[None], kv[None])[0, 0]
        Err = torch.linalg.solve_triangular(A, err0, upper=False, unitriangular=True)
        upd = kv[..., None] * (beta[:, None] * Err)[:, None, :]             # T-1, active, d
        self.M = self.M.reshape(U, -1).index_add(0, ki.reshape(-1), upd.reshape(-1, upd.size(-1))).view_as(self.M)
        self.ms.append((time.perf_counter() - t0) * 1000)

    def memory_bytes(self) -> int:
        return self.M.numel() * 4


class FineTune(LLMLearner):
    """The gradient baseline: a copy of the LLM, one Adam step per session (loss × reward)."""

    def __init__(self, model, tok, lr: float):
        super().__init__(copy.deepcopy(model), tok)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def learn(self, s: data.Session):
        tk = Tokens(self.tok, [s.text])
        with torch.enable_grad():
            lg = self.model(input_ids=tk.ids).logits.float()[0]
            loss = F.cross_entropy(lg[:-1], tk.ids[0, 1:])
            self.opt.zero_grad(set_to_none=True)
            (loss * s.reward).backward()
            self.opt.step()


class RAG(Retrieval, LLMLearner):
    """Keep every sentence; put the best match before the question; the LLM reads it (lifetest.Retrieval)."""

    def __init__(self, model, tok):
        Retrieval.__init__(self, model)
        self.tok = tok


@torch.no_grad()
def loss_ref(model, tok, stories: list[str]) -> float:
    """The LLM's average loss per token (nats) on held-out stories: surprise = loss / this."""
    tk = Tokens(tok, [data._fit(s, data.SESSION_BYTES) for s in stories])
    tot, cnt = 0.0, 0
    for i in range(0, len(stories), CHUNK):
        lg = model(input_ids=tk.ids[i:i + CHUNK], attention_mask=tk.mask[i:i + CHUNK]).logits.float()
        y, m = tk.ids[i:i + CHUNK, 1:], tk.mask[i:i + CHUNK, 1:]
        nll = F.cross_entropy(lg[:, :-1].flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        tot, cnt = tot + float((nll * m).sum()), cnt + int(m.sum())
    return tot / cnt


# -- the dev choice and the verdict (docs/HYBRID.md) ----------------------------------------------------------------
def choose_hybrid(rows: dict) -> str:
    """Among settings with bits per byte risen ≤ 5%: the highest dev recall (3 readings) + recall in other words;
    ties go to the smaller rise. If none qualifies, the smallest rise."""
    ok = {k: v for k, v in rows.items() if v["bpb_rise"] <= 0.05}
    if not ok:
        return min(rows, key=lambda k: rows[k]["bpb_rise"])
    return max(ok, key=lambda k: (ok[k]["recall_main"] + ok[k]["recall_paraphrase"], -ok[k]["bpb_rise"]))


def verdict(test: dict) -> dict:
    h = test["hybrid"]
    rules = {
        "H1_learns": h["recall_main"] >= 0.80,
        "H2_no_forgetting": h["retention"] >= 0.80 and h["bpb_rise"] <= 0.05,
        "H3_beats_finetuning": bootstrap_lower(h["main_correct"], test["finetune"]["main_correct"]) > 0,
        "H4_matches_rag": bootstrap_lower(h["main_correct"], test["rag"]["main_correct"]) > -0.05,
        "H5_other_words": h["recall_heldout"] >= 0.60,
    }
    detail = {k: bootstrap_lower(h["main_correct"], test[k]["main_correct"]) for k in ("finetune", "rag")}
    return {"rules": rules, "recall_diff_lower95": detail, "pass": all(rules.values())}


def _streams(args):
    sp = data.splits(args.data_dir, need_train=False)
    sizes = data.SMALL_LIFE if args.small else data.FULL_LIFE
    streams = {"dev": data.life_stream(sp.life_dev, sp.names["dev"], SEEDS["dev"], sizes),
               "test": data.life_stream(sp.life_test, sp.names["test"], SEEDS["test"], sizes)}
    streams["test"].heldout = True
    return sp, streams


def _live(learner, stream, val) -> dict:
    out = live(learner, stream, val)
    if isinstance(learner, Hybrid):
        out["ms_per_session_median"] = float(np.median(learner.ms))
        out["memory_bytes"] = learner.memory_bytes()
    return out


def _line(name, r) -> str:
    keys = ("recall_main", "recall_once", "recall_reward", "change_new_wins", "recall_paraphrase", "recall_heldout",
            "retention", "bpb_rise")
    return name + ": " + " ".join(f"{k.replace('recall_', '')} {r[k]:+.3f}" if k == "bpb_rise" else
                                  f"{k.replace('recall_', '')} {r[k]:.2f}" for k in keys if k in r)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["dev", "test"], required=True)
    p.add_argument("--llm", default=LLM)
    p.add_argument("--data-dir", default="brain/out/data")
    p.add_argument("--small", action="store_true", help="a short life (a quick check of the code, dev only)")
    p.add_argument("--val-stories", type=int, default=300)
    p.add_argument("--dev-report", help="the dev stage's output (test stage)")
    p.add_argument("--only", help="dev stage: run only these comma-separated learners (hybrid, finetune)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    sp, streams = _streams(args)
    val = sp.val[: args.val_stories]
    familiar = sp.train[:20000]
    tok, model = load_llm(args.llm)
    ref = loss_ref(model, tok, val)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out.read_text()) if out.exists() else {}
    report.setdefault("setup", {}).update(llm=args.llm, small=args.small, loss_ref=ref, grid=GRID, lr_grid=LR_GRID,
                                          sessions={k: len(s.sessions) for k, s in streams.items()})
    addresses: dict[float, Brain] = {}

    def address(bag):
        if bag not in addresses:
            addresses[bag] = make_address(bag, familiar)
        return addresses[bag]

    def save():
        out.write_text(json.dumps(report, indent=1, default=float))

    if args.stage == "dev":
        only = set(args.only.split(",")) if args.only else {"hybrid", "finetune"}
        dev = report.setdefault("dev", {})
        if "hybrid" in only:
            rows = dev.setdefault("hybrid", {})
            for bag in GRID["key_bag"]:
                for rate in GRID["rate"]:
                    for gain in GRID["gain"]:
                        key = f"bag={bag} rate={rate} gain={gain}"
                        if key in rows:
                            continue
                        r = _live(Hybrid(model, tok, address(bag), rate, gain, ref), streams["dev"], val)
                        r.pop("main_correct")
                        rows[key] = r
                        print(_line(f"dev hybrid {key}", r), flush=True)
                        save()
            report["chosen_hybrid"] = choose_hybrid(rows)
        if "finetune" in only:
            rows = dev.setdefault("finetune", {})
            for lr in LR_GRID:
                if str(lr) in rows:
                    continue
                r = _live(FineTune(model, tok, lr), streams["dev"], val)
                r.pop("main_correct")
                rows[str(lr)] = r
                print(_line(f"dev finetune {lr}", r), flush=True)
                save()
            report["chosen_finetune"] = choose(rows)
        if "hybrid" in only:
            for name, learner in (("rag", RAG(model, tok)), ("lookup", Lookup(model))):
                r = _live(learner, streams["dev"], val)
                r.pop("main_correct")
                dev[name] = r
                print(_line(f"dev {name}", r), flush=True)
        save()
        print("chosen:", report.get("chosen_hybrid"), report.get("chosen_finetune"))
        return

    devrep = json.loads(Path(args.dev_report).read_text())
    ch = devrep["chosen_hybrid"]
    bag, rate, gain = (float(x.split("=")[1]) for x in ch.split())
    lr = float(devrep["chosen_finetune"])
    report["chosen"] = {"hybrid": ch, "finetune": lr}
    test = report.setdefault("test", {})
    learners = {
        "hybrid": lambda: Hybrid(model, tok, address(bag), rate, gain, ref),
        "rag": lambda: RAG(model, tok),
        "lookup": lambda: Lookup(model),
        "finetune": lambda: FineTune(model, tok, lr),
        "no_learning": lambda: LLMLearner(model, tok),
    }
    for name, make in learners.items():
        if name in test:
            continue
        test[name] = _live(make(), streams["test"], val)
        test[name]["main_correct"] = [bool(x) for x in test[name]["main_correct"]]
        print(_line(f"TEST {name}", test[name]), flush=True)
        save()
    report["resources"] = {"peak_rss_mb": peak_rss_mb()}
    report["verdict"] = verdict(test)
    save()
    print(json.dumps(report["verdict"], indent=1))


if __name__ == "__main__":
    main()
