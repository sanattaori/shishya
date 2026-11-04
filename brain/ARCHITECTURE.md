# Shishya Brain: architecture (v3)

> This page covers v3 only. For the current model, start with [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

*Written 1 Nov 2026, for the v3 model judged in [RESULTS.md](RESULTS.md) section 6. The plan and the pass rules
are in [docs/PLAN.md](../docs/PLAN.md). What using it looks like: [DEMO.md](DEMO.md).*

Shishya Brain is a small language model that **keeps learning after training**. It is born with backprop and
lives with a local rule:

- **Childhood:** ordinary training on children's stories, read as bytes. It learns language, and it learns how to
  use a memory that it writes itself.
- **Life:** no more gradients and no optimiser. After each text it reads, a local rule changes its **lasting
  synapses**, and the change stays. Nothing else in the model changes.

**Where v3 stands.** It passes the learning rules (L1–L3): 99% of facts recalled after 3 readings, 96% after one,
95% of the earliest facts kept, against 21–22% for GPT and BDH fine-tuned on every session. It fails L4: 665 ms
per session on a CPU, where the rule asks for 50 ms. So the overall verdict is FAIL.

## The picture

```text
                       one 256-byte text, read byte by byte
                                      │
                               byte embedding (256 → 256)
                                      │
      ┌───────────────────────────────▼────────────────────────────────┐
      │  layer l (6 of them)                                           │
      │                                                                │
      │   x ──► sparse neurons ──► fast synapses (this text only) ─┐   │
      │   │      ReLU(x·E)          causal linear attention         │   │
      │   │      4 heads × 4,096                                    + ──► LayerNorm ──► gate ──► x'
      │   │                                                         │   │
      │   │   address k_t ──► lasting memory M_l ──► read · W_out ──┘   │
      │   │   (fixed, sparse:    16,384 × 64                            │
      │   │    40 of 16,384)     kept between texts                     │
      │   │                         ▲                                   │
      │   └─► value v_t = LN(x·W_in) ┘ written after the text           │
      └───────────────────────────────┬────────────────────────────────┘
                                      │
                              next-byte probabilities
                                      │
       after the text: surprise per byte ──► delta rule ──► new M_1 … M_6
```

## 1. The core: BDH

The core is Pathway's BDH ("the Dragon Hatchling", MIT licence), unchanged. `brain/tests` checks that our code
gives exactly the same logits as `third_party/bdh.py`.

| | `bee` size (the one judged) |
|---|---|
| Input | bytes (vocabulary 256), up to 256 per text |
| Layers | 6, all sharing one set of weights (as in BDH) |
| Width | 256 |
| Neurons | 4 heads × 4,096 = 16,384 per layer, ReLU, so most are zero |
| Parameters | 12,713,984 for the core; 12,795,916 with the memory's read path |

One layer, in `Brain.forward` (`models.py`):

1. **Sparse neurons:** `x_sparse = ReLU(x · E)`.
2. **Fast synapses:** a causal linear attention over this text, with BDH's rotary phases. This is BDH's own
   working memory. It is Hebbian (neurons that fire together link), and it is gone when the text ends.
3. **Lasting memory read** (ours, section 2): added to the fast-synapse output before its LayerNorm.
4. **Gate and output:** `y_sparse = ReLU(yKV · E_v)`, `x' = LN(x + LN((x_sparse ⊙ y_sparse) · D))`.

## 2. The lasting memory

**Shape.** One matrix per layer, each 16,384 units × 64 numbers. All six layers share one address. In all:
6 × 16,384 × 64 floats, about **25 MB**. The size is fixed: it never grows, however much the Brain reads.

**Address (key) `k_t`.** At each byte, a sparse code: 40 of the 16,384 units are active, and the code has unit
length. The address is fixed. It is never trained (section 2.1).

**Value `v_t`.** What the layer saw: `v_t = LayerNorm(x_t · W_in)`, 64 numbers. `W_in` was learned in childhood.

**Read.** `read_t = k_t · M_l`, the sum of the 40 active rows. Then `read_t · W_out` (one per head), times a
learned gain per layer, is added to the fast-synapse output. A read costs 40 row lookups per byte per layer.

**What is stored where.** The rule links the address at byte *t* to the value at byte *t + 1* (section 3). So when
the same address comes again, the memory hands the network "what came next last time". Asked "Naros's cat is
called", the address of that sentence brings back the start of "Fugez", and the network writes the rest.

### 2.1 The address, v1 to v3

The address is where the versions differ. It decides which memories share synapses, and so which ones overwrite
each other.

| Version | Address | Two facts of the same kind alike | Result |
|---|---|---:|---|
| v1 | BDH's own sparse neurons | not measured | recall 35% |
| v2 | the last 32 bytes, each a random ±1 code bound to its place, fading 0.9 per byte, top 40 of 16,384 | 61% | recall 67%; old facts lost |
| **v3** | **the words of the current sentence (up to 8), each counting by how new it is** | **13%** | **recall 99%; 95% of early facts kept** |

### 2.2 The v3 address in detail (`Brain.word_code`)

1. **Words, not bytes.** A word is a run of letters. Any other byte except a space is a word of its own
   ("Zubak's" is `Zubak`, `'`, `s`). A word's code is the sum of its bytes' random ±1 codes, each bound to its
   place in the word, divided by √length.
2. **Slots.** The word being read so far goes in slot 0. The up to 7 words before it go in slots 1–7. Each is
   bound to its slot's own random code, and every slot counts the same (no fading). A name four words back counts
   as much as the word just read.
3. **The sentence.** Words before the last `.`, `!`, `?` or new line are left out. A fact inside a story and the
   same words asked on their own then get exactly the same address. (In the brain: event boundaries.)
4. **Novelty.** Each earlier word is scaled by how new it is:

   ```text
   novelty(word) = log((1 + N) / (1 + count(word))) / log(1 + N)
   ```

   The counts come from the 4.0M words of the first 20,000 childhood stories. They are hashed into 2²⁰ buckets,
   counted once, then fixed. A made-up name counts 1.0, "secret" 0.6, "called" 0.52, "is" 0.38, "the" 0.22.
   (In the brain: the novelty signal.)
5. **Sparse.** Keep the 40 strongest of the 16,384 units, ReLU, unit length.

The result: "Naros's cat is called" and "Pimev's cat is called" share 13% of their address, not v2's 61%. So
writing one barely touches the other.

## 3. The write rule (life)

After the Brain reads a text, `Brain.write` stores it. Every synapse changes only from what its own two neurons
did, plus two global signals:

```text
ΔM[i, ·] = β_t · k_t[i] · (v_{t+1} − k_t · M)          β_t = min(rate_l · life_rate · surprise_t · reward, 1)
```

- **Pre before post:** the address at *t* is linked to the value at *t + 1*. The memory learns what follows what.
- **Error-correcting (the delta rule):** it writes only the part of `v_{t+1}` that the memory doesn't already
  predict. What is known is barely written again. A changed fact overwrites the old one.
- **Surprise:** this byte's loss divided by the average loss from childhood, capped at 4. A new name is stored
  strongly; "Once upon a time" hardly at all.
- **Reward:** 1 for normal reading, 3 for a correction, so a correction is stored 3 times as strongly.
- **Rates:** `rate_l` = sigmoid of a number per layer, learned in childhood. `life_rate` = 0.5, chosen on the dev
  stream from {0.5, 1, 2, 4}.
- **Sparse:** each byte touches only the 40 rows of its active units.

The rule is applied exactly as if byte after byte, but computed in one go per text: one triangular solve over the
text's bytes, then a scatter-add into the active rows (Yang et al., 2024).

**Forget** (`Brain.forget`): the same rule with 0 as the value, at full strength, 8 passes. Afterwards the text's
addresses read back almost nothing, and the model answers as if it had never read it. Memories that share those
addresses lose a little too. This is Larimar's forgetting, done with the local rule.

## 4. Childhood: learning how to learn (`childhood.py`)

- **Bytes:** 100M bytes of TinyStories, the same bytes for GPT, BDH and the Brain.
- **Episodes:** 4 texts of 256 bytes in a row. The Brain reads them in order and writes each into its memory
  before the next. Backprop runs *through* those writes. So childhood trains the read path (`W_in`, `W_out`,
  gains), the write rates, and the core's way of using what the memory hands back.
- **Memory drills:** 30% of episodes put 1–3 made-up facts in the first text and state them again in the last.
  Drill names come from a training list. The life test uses other names, never seen in childhood. The drills use
  the same 5 kinds of fact sentence as the life test.
- **Mixed ages:** each batch slot keeps its memory across training steps, and is emptied with chance 1/64. So
  the Brain learns to read memories that are fresh and memories that are full.
- **The surprise scale:** `loss_ref`, a running average of the training loss.

**v3 had no childhood of its own.** It is v2's trained network, unchanged, with v3's address
(`brain/make_v3.py`). Only the address changed, and the address is fixed code. The novelty counts are made once
from childhood text.

## 5. What is learned, what is fixed, what changes

| Part | Size | Learned in childhood | Changes in life |
|---|---:|:---:|:---:|
| BDH core (embedding, E, E_v, D, output) | 12.7M | yes | no |
| Memory read path (`W_in`, `W_out`, gains, rates) | 82K | yes | no |
| Address codes (random ±1, from a fixed seed) | | no | no |
| Word counts for novelty | 1M buckets | counted once | no |
| **Lasting memory M** | **6.3M numbers (25 MB)** | **written in drills** | **yes, by the local rule only** |

The memory is an explicit state, kept outside the model. It can be saved, copied, emptied and measured.

## 6. How it is judged

The life test (`lifetest.py`, rules in [docs/PLAN.md](../docs/PLAN.md)):

- About 520 sessions of up to 256 bytes: unseen stories, half of them with up to 2 made-up facts.
- 100 main facts seen 3 times, 50 seen once, 50 seen once as a correction (reward 3), 30 that change later.
- Questions are asked in a **fresh context**, with nothing before them: "Zib's cat is called" → which of 5 names
  is most likely (chance 20%).
- **Baselines:** GPT and plain BDH of the same size, the same childhood bytes, one Adam step on every session, at
  the learning rate chosen on a dev stream.
- **Pass rules L1–L4**, fixed before any run: learns (≥ 80% after 3 readings), doesn't forget, beats gradient
  learning, small machine (≤ 2 GB, ≤ 50 ms per session on a CPU).
- **Dev and test are separate.** All choices were made on the dev stream, with dev names. Each version's rules
  were committed before its test run.

## 7. Costs

| | v3 |
|---|---|
| Memory on a CPU, whole life test | 1.2 GB peak (rule: ≤ 2 GB) |
| One 256-byte session, CPU (Kaggle) | 665 ms (rule: ≤ 50 ms) |
| One session, T4 GPU | 38 ms |
| Childhood speed, T4 | 4,511 bytes/s (GPT: 62,058) |

The time goes to the BDH core, about 50 billion operations per session (estimated), not to the memory. Each
memory read or write touches only 40 rows per byte per layer.

## 8. Known limits

- **Questions in other words: 34%** (chance 20%). The address is made of the question's surface words, so "The
  town where Pimev lives is called" does not find "Pimev lives in a town called". This is the main weak spot.
- **The address is hand-built.** Words, sentence boundaries and novelty are fixed code, close in spirit to a
  keyword index. How much the network's own learning adds on top of a plain lookup has not been measured yet.
- **No retrieval baseline yet** ("keep every text, look it up, let GPT read it"). That is how LLMs usually learn
  after training, so it is the comparison that matters most.
- **A narrow test:** children's stories, 12.8M parameters, 5 kinds of fact sentence (the same 5 in childhood
  drills and the test), 260 facts, 520 sessions, and 1-in-5 choice. Capacity (thousands of facts in the same
  25 MB) and facts that must be combined are untested.
- **Changed facts:** the new name wins 77%. Novelty + metaplasticity + reconsolidation got 93% on dev, but it is
  not part of v3.
- **Speed:** fails L4 on a CPU.

## 9. Code map

| File | What is in it |
|---|---|
| `models.py` | `Brain` (BDH + lasting memory): `forward`, `word_code` (v3 address), `novelty`, `learn_familiarity`, `surprise`, `write`, `_write_window` (sparse delta rule, metaplasticity), `forget`. `GPT`. `V3` settings. `build` |
| `data.py` | TinyStories bytes, made-up names split into train / dev / test, fact sentences (`RELATIONS`), childhood episodes, the life stream |
| `childhood.py` | training one model; backprop through the memory writes |
| `make_v3.py` | v2's network + v3's address → the v3 model |
| `lifetest.py` | the life test, the learners (`Hebbian`, `Gradient`), the verdict |
| `study.py` | one dev life, recording when each fact is learned and lost; `--set` to try a setting |
| `probes.py` | how alike addresses are; whether a question finds its own fact |
| `demo.py` | write, read, recall, correct, forget ([DEMO.md](DEMO.md)) |
| `third_party/bdh.py` | Pathway's BDH, vendored unchanged |
| `kaggle/` | the full runs on Kaggle's free GPU |

## 10. Try it

The trained models are not in git. They are in the output of the private Kaggle kernel
`sanattaori/shishya-brain-life2`. With access to it:

```bash
.venv/bin/pip install torch numpy kaggle
kaggle kernels output sanattaori/shishya-brain-life2 -p brain/out/kaggle-shishya-brain-life2
mkdir -p brain/out/runs brain/out/data
ln -sfn ../kaggle-shishya-brain-life2/out/brain2 brain/out/runs/brain2
mv brain/out/kaggle-shishya-brain-life2/data/* brain/out/data/
.venv/bin/python -m brain.make_v3 brain/out/runs/brain2 brain/out/runs/brain3
.venv/bin/python -m brain.demo brain/out/runs/brain3            # about a minute on a laptop CPU
```
