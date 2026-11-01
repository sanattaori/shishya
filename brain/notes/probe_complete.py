# A one-off probe from 1 Nov 2026 (v5 study), kept so the result in RESULTS.md section 9 can be rerun:
#   python brain/notes/probe_complete.py 512 100
"""Cue completion probe (school world, dev names): can a partial address be completed into its fact's full address?

The memory also stores, by the delta rule, a fingerprint of each address (a fixed random projection to r numbers).
A question reads back a fingerprint (a blend of the stored ones, weighted by overlap), turns it back into an
address (top 40 of the inverse projection), and we measure how close that is to its own fact's address.
"""
import random
import sys

import torch

torch.set_num_threads(1)
from brain import data, school  # noqa: E402
from brain.models import Brain, BrainConfig  # noqa: E402

R_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_STORIES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
sp = data.splits("brain/out/data", need_train=False)
world = school.make_world(sp.names["dev"], 40)
facts = list(world.facts.values())
rng = random.Random(0)
b = Brain(BrainConfig(n_layer=1, n_embd=32, n_head=2, mult=8, mem_dim=8, key="words", key_window=8, key_decay=1.0,
                      key_novelty=True, key_bag=4.0))
b.learn_familiarity(sp.train[:20000])
U, A = b.cfg.key_size, b.cfg.key_active


def keys_of(text):
    kidx, kval = b.window_keys(torch.tensor([list(text.encode()[:256])]))
    return kidx[0], kval[0]


# the stream of writes: each lesson (in a story), with ordinary stories between; every step is written
writes, own_pos = [], {}
for i, f in enumerate(facts):
    sent = f.fill(school.WORDINGS[f.relation][0][0])
    text = data.with_facts(rng.choice(sp.life_dev), [sent], 257)
    kidx, kval = keys_of(text)
    p = text.encode().index(sent.encode()) + len(sent.encode()) - len(f.obj.encode()) - 2
    for t in range(kidx.size(0)):
        if t == p:
            own_pos[i] = len(writes)
        writes.append((kidx[t], kval[t]))
    if i < N_STORIES:
        ki, kv = keys_of(data._fit(rng.choice(sp.life_dev), 256))
        writes += list(zip(ki, kv))
print(f"{len(writes)} writes, sketch {R_DIM}", flush=True)
g = torch.Generator().manual_seed(7)
R = torch.randn(U, R_DIM, generator=g) / R_DIM ** 0.5
M = torch.zeros(U, R_DIM)
for kidx, kval in writes:                       # the delta rule, β = 1: k → fingerprint(k)
    target = kval @ R[kidx]                     # r
    err = target - kval @ M[kidx]
    M[kidx] += kval[:, None] * err[None]


def dense(kidx, kval):
    return torch.zeros(U).scatter_(0, kidx, kval)


def complete(kidx, kval):
    s = kval @ M[kidx]                          # the fingerprint read back
    pre = R @ s                                 # back to an address
    top = pre.topk(A)
    v = torch.relu(top.values)
    return top.indices, v / (v.norm() + 1e-6)


own_keys = torch.stack([dense(*writes[own_pos[i]]) for i in range(len(facts))])
res = {"question": [], "completed": [], "mixed": []}
found = {k: 0 for k in res}
for i, f in enumerate(facts):
    q = f.fill(school.WORDINGS[f.relation][1][0])
    kidx, kval = keys_of(q + " " + f.obj + ".")
    pos = len(q.encode())
    kq = dense(kidx[pos], kval[pos])
    kc = dense(*complete(kidx[pos], kval[pos]))
    km = (kq + kc) / (kq + kc).norm()
    for name, k in (("question", kq), ("completed", kc), ("mixed", km)):
        sims = own_keys @ k
        res[name].append(float(sims[i]))
        found[name] += int(sims.argmax() == i)
for name in res:
    share = sum(res[name]) / len(res[name])
    print(f"{name:10s}: share of own address {share:.2f}, finds own {found[name] / len(facts):.0%}", flush=True)
