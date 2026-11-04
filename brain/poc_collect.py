"""One-hour POC (docs/V8_POC.md): take a hard school with v7 as it is, and save what any read design needs.

It learns exactly as v7 does, so its scores are a control against v7's own. It also saves the write traces and, for
each exam option, the frozen reader's logits and the address at each scored step. Read designs can then be tried in
seconds, with no reader runs. No text is kept, only numbers.

    python -m brain.poc_collect changes --names dev --out brain/out/poc/changes_dev.pt
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from . import v5, v7
from .hardschool import changes, dense
from .hybrid import run_llm
from .school import make_world

RUN = "brain/out/kaggle-shishya-brain-v7/out/v7/l3"
BDH = "bdh:brain/out/hf/bdh100m/bdh-100m-bytes.pt"


class Collect(v7.Sleeper):
    """v7 with no sleep; ``loglik`` is v5's, and it also saves each option's scored steps."""

    def __init__(self, base):
        super().__init__(base, mode="none")
        self.exams: list[list[dict]] = []
        self.lessons: list[str] = []

    def learn(self, session):
        self.lessons.extend(session.facts)
        return super().learn(session)

    @torch.no_grad()
    def loglik(self, prompts, conts):
        texts = [p + c for p, c in zip(prompts, conts)]
        ps = v5.Parsed(self.tok, self.who, texts)
        self.meaning(ps, want_logits=True)                # a one-sentence text read alone is the text: one reader run
        tk = ps.tk
        scores = torch.zeros(len(texts))
        items = []
        rows = {}
        for i, pos in ps.groups:
            rows.setdefault(i, []).append(pos)
        for n in range(len(texts)):
            L = tk.lens[n]
            if len(rows.get(n, [])) == 1 and rows[n][0] == list(range(1, L)):
                base = torch.zeros(L, ps.iso_logits[(n, 1)].size(-1))
                base[1:] = ps.iso_logits[(n, 1)]
            else:
                base = run_llm(self.model, tk.ids[n:n + 1, :L], tk.mask[n:n + 1, :L])[0]
            bias = torch.zeros_like(base)
            g = ps.gate[n, :L]
            if g.any():
                r = self.read(ps.kidx[n, :L][g], ps.kval[n, :L][g], ps.what[n, :L][g])
                bias[g] = float(self.reader.gain) * (r @ self.Z.T)
            lp = F.log_softmax(base + bias, dim=-1)
            first = next(k for k, c in enumerate(tk.char_starts[n]) if k > 0 and c >= len(prompts[n]))
            a, b = first - 1, L - 1
            tgt = tk.ids[n, first:L]
            scores[n] = lp[a:b].gather(1, tgt[:, None]).sum()
            items.append({"logits": base[a:b].detach().cpu().clone().half(), "tgt": tgt.cpu().clone(),
                          "gate": ps.gate[n, a:b].clone(), "kidx": ps.kidx[n, a:b].to(torch.int16).clone(),
                          "kval": ps.kval[n, a:b].clone(), "what": ps.what[n, a:b].clone(),
                          "prompt": prompts[n], "cont": conts[n]})
        self.exams.append(items)
        return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=["changes", "dense"])
    p.add_argument("--names", choices=["dev", "test"], default="dev")
    p.add_argument("--people", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, help="the world's seed; by default 0 for dev names, 1 for test names")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    st, sp, _, _ = v5.load_student(RUN, BDH, "brain/out/data", "cpu")
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    c = Collect(st)
    seed = a.seed if a.seed is not None else (0 if a.names == "dev" else 1)
    w = make_world(sp.names[a.names], a.people, seed=seed)
    stories = sp.life_dev if a.names == "dev" else sp.life_test
    res = changes(c, w, stories, sp.names[a.names], seed) if a.test == "changes" else dense(c, w, stories, seed)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("correct")}), flush=True)
    torch.save({"res": res, "sentences": c.lessons, "exams": c.exams,
                "K_i": torch.stack(c.K_i), "K_v": torch.stack(c.K_v),
                "Wh": torch.stack(c.Wh), "tokens": torch.tensor(c.tokens), "beta": torch.tensor(c.beta),
                "fact": torch.tensor(c.fact), "gain": float(c.reader.gain.detach()), "Z": c.Z.detach(),
                "M_rows": c.stats()}, a.out)
    print(f"DONE {time.time() - t0:.0f}s traces {len(c.tokens)}", flush=True)


if __name__ == "__main__":
    main()
