# Shishya Brain: results

**Status, 31 Oct 2026, 17:15 UTC: both verdicts are in (section 3). v1 FAIL; v2 FAIL on L1, L2 and L4, passes L3 by a wide margin.** Sections 1–2 are checks, not verdicts. The pass
rules are in [docs/PLAN.md](../docs/PLAN.md) and did not change. Sections 1 and 2 are checks of the code and of the
design, run on this CPU (tiny models, trained on part of the TinyStories *validation* file) and in a 15-minute Kaggle
canary.

## 1. What the CPU checks showed

### Tiny models, 2.5M bytes each (≈0.9M parameters)

| Model | Held-out bits/byte |
|---|---:|
| GPT | 3.046 |
| BDH | 1.837 |
| Brain v1 | 2.197 (read with an empty memory) |

- BDH learns much faster per byte than GPT at this size.
- In the short life test (12 facts), no model learned the facts. All recall was within noise of chance.
- Written with the code to see that it works end to end. Not a result.

### Brain v1, 15M bytes: does the memory help at all?

| Check (dev names only) | Result |
|---|---|
| Held-out bits/byte | 1.102 |
| A made-up name, two texts after it was stated (`brain/diagnose.py`) | 3.211 bits with an empty memory → 3.190 with its memory: **almost no help** |
| The same text read twice | 1.08 → 0.98 bits/byte; forcing full-strength writes: still 0.98 |
| How alike the addresses of different steps are (mean cosine, by layer) | 0.59–0.82; a third of all neurons fire on every step |

**Address probe** (`python -m brain.probes model …`): does a bare question find its own fact's address among 100?

| Layer | v1 finds its fact (chance 1%) |
|---|---:|
| 0 | 1% |
| 1 | 8% |
| 2 | 21% |
| 3 | 25% |

At the step before the name, the addresses of different facts are 0.94–1.00 alike. The memory can't tell whose fact
it is. This led to the v2 amendment.

### v2's window code, untrained (`python -m brain.probes window`)

Settings: 16,384 units, top 40, 100 dev facts.

| Window (bytes) | Fading 1.0 | 0.95 | 0.9 |
|---:|---:|---:|---:|
| 16 | 28% | 28% | 28% |
| 24 | 83% | 83% | 83% |
| 32 | 95% | 99% | **97%** (v2's setting) |
| 40 | 71% | 88% | 97% |
| 48 | 53% | 89% | 95% |

- The earlier quick test with 2,048 units gave 96% for the same setting.
- A window too short doesn't reach the name ("…lives in a town called" is 27 bytes). A window too long, without
  fading, is drowned by the story around the fact.

### Tiny v2, 15M bytes: the memory now works (on short gaps)

Same recipe and bytes as the tiny v1 Brain above.

| Check (dev names only) | v1 | **v2** |
|---|---:|---:|
| Held-out bits/byte | 1.102 | **0.997** |
| A made-up name two texts after it was stated: bits/byte with an empty memory | 3.211 | 3.750 |
| … with its own memory | 3.190 | **0.341** |
| **What the memory saves** | 0.02 bits | **3.41 bits** |

For reference, the tiny GPT and BDH (trained on 2.5M bytes only) give 5.33 and 4.46 on the same names.

In plain words: v2 recalls the made-up names almost perfectly two texts later; v1 didn't. This is a short-gap
drill, like childhood. It is not the life test, where facts must survive about 500 sessions and are asked about in
a fresh context.

**The short CPU life test** (76 sessions, 12 main facts, so noisy; the tiny GPT and BDH here had 6× less training,
so only the Brain's own numbers mean much):

| | Brain v2 (local rule) | GPT + fine-tuning | BDH + fine-tuning | Brain v2, no learning |
|---|---:|---:|---:|---:|
| Recalled after 3 exposures | **58%** (7 of 12) | 33% | 42% | 25% |
| After 1 exposure | 83% (5 of 6) | 17% | 33% | |
| After 1 correction (reward) | 83% | 17% | 50% | |
| Changed fact: the new name wins | 100% (4 of 4) | 75% | 25% | |
| Early facts kept to the end | 3 of 4 | | | |
| Held-out bits/byte over the life | **−1.8%** (better) | +5.4% | +0.5% | |
| Asked in other words (reported only) | 17%: no better than chance, as predicted | | | |

- The dev choice of write rate made no difference: 67% at every rate. The writes are already at full strength,
  capped at 1.
- On this CPU the tiny v2 takes 77 ms per session at a peak of 519 MB. L4's 50 ms fails even for the tiny model.

**A measurement bug, fixed in e3a29b4.** The CPU check ran in a child process and read its peak memory
from `ru_maxrss`. On Linux that also counts the parent's memory at the fork. The first reading here was 2.7 GB, and
the canary's 5.5 GB below was inflated the same way. The check now reads the child's own peak (`VmHWM`).

## 2. Kaggle canary (2× T4, 20 steps per model, full `bee` size)

| | GPT | BDH | Brain v1 |
|---|---:|---:|---:|
| Training speed (bytes/s) | 45,176 | 6,864 | 3,162 |
| Parameters | 12,706,304 | 12,713,984 | 12,845,068 |

On a CPU (Kaggle's), the `bee` Brain v1 takes **625 ms** per 256-byte session. Rule L4 says ≤ 50 ms, so L4 fails as
written. The 5.5 GB peak memory measured there is wrong. Part of it was the evaluation scoring 256 questions at once
(fixed in 0484c13). Part of it was the parent's memory counted by `ru_maxrss` (see above). v1's memory is
re-measured with the fixed code after its run.

## 3. The verdict runs

### v1 (Kaggle `sanattaori/shishya-brain-life`, code ba36374, 2× T4, finished 15:51 UTC): **FAIL**

#### Childhood (same bytes for every model)

| Model | Parameters | Bytes | Held-out bits/byte | Hours | Bytes/s |
|---|---:|---:|---:|---:|---:|
| gpt | 12,706,304 | 100,007,936 | 0.588 | 0.4 | 62,058 |
| bdh | 12,713,984 | 100,007,936 | 0.532 | 3.4 | 8,240 |
| brain | 12,845,068 | 100,007,936 | 0.561 | 7.8 | 3,548 |

Stage 1 (within 5% of GPT): BDH -9.5%, Brain -4.5%.

#### Life (test stream, fresh-context questions, chance = 20%)

| Learner | Setting (dev) | Recall, 3 exposures | 1 exposure | 1 correction (reward 3) | Changed fact: new wins | Paraphrased | Early facts kept | Bits/byte change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Shishya Brain** (local rule only) | 0.5 | 35% | 24% | 24% | 43% | 25% | 0.05 of 0.80 | -5.5% |
| GPT + fine-tuning | 3e-05 | 21% | 14% | 20% | 30% | 22% | 0.15 of 0.45 | +1.9% |
| BDH + fine-tuning | 3e-05 | 22% | 16% | 18% | 30% | 20% | 0.15 of 0.35 | +1.8% |
| No learning (as after childhood) | | brain 21% / gpt 17% / bdh 25% | | | | | | |

#### Verdict (rules fixed in docs/PLAN.md before the run)

| Rule | Result |
|---|---|
| L1_learns | **fail** |
| L2_no_forgetting | **fail** |
| L4_small_machine | **fail** |
| L3_beats_gradient | pass |

L3 detail: vs gpt: recall difference lower 95% bound +0.03, vs bdh: recall difference lower 95% bound +0.00
L4 measured: 562 ms per 256-byte write, peak 9208 MB.

**Overall: FAIL**


The L4 memory figure (9.2 GB) is wrong: it is the measurement bug from section 1 (the parent's memory counted).

| | Kaggle's CPU (old code) | This 4-core CPU (fixed code, same test life) |
|---|---:|---:|
| Peak memory | 9.2 GB (wrong) | **996 MB** (L4's 2 GB: OK) |
| Speed per 256-byte session | 562 ms | 373 ms |

So L4 fails on speed only: the 50 ms target is about 7–11× too fast for BDH at this size.

**What happened, in plain words**

- **Childhood worked for all three.** BDH reads held-out text 9.5% better than GPT at the same size and bytes, as
  Pathway reports. The Brain is 4.5% better than GPT.
- **The Brain learned facts by itself, then lost them.**
  - Learned: 80% of the 20 earliest facts were recalled at the 30% mark, from its local rule alone.
  - Lost: at the end, only 5% were. Later writes overwrote them.
  - This is the address problem found on the CPU (section 1): different facts' addresses are too alike. It is what
    v2 changes.
- **Its reading never suffered from its memory:** held-out bits/byte 5.5% better after the life.
- **It beat learning by gradients (L3 passes), but far from the 80% L1 asks.** 35 of 100 facts, against 21 (GPT)
  and 22 (BDH).
  - Against GPT, by higher recall: the 95% lower bound is +3 points.
  - Against BDH, by the rule's other branch: the recall lower bound is exactly 0. But it forgets less: bits/byte
    −5.5% against +1.8%, a gap of 7.3 points where 2 are needed, and recall is not lower.
- The test life had 636 sessions (the plan said "about 520").
- **Gradient learning shows the classic trade-off.** The dev runs: rates high enough to learn more facts destroy
  the model's English. So the pre-registered rule picked the gentlest rate, which barely learns.

| Model | Learning rate | Dev recall, 3 exposures | Held-out bits/byte change |
|---|---:|---:|---:|
| gpt | 3e-05 | 20% | +2% |
| gpt | 0.0001 | 22% | +12% |
| gpt | 0.0003 | 23% | +48% |
| gpt | 0.001 | 23% | +153% |
| gpt | 0.003 | 17% | +334% |
| bdh | 3e-05 | 20% | +2% |
| bdh | 0.0001 | 22% | +11% |
| bdh | 0.0003 | 27% | +44% |
| bdh | 0.001 | 23% | +118% |
| bdh | 0.003 | 21% | +204% |

### v2 (Kaggle `sanattaori/shishya-brain-v2` training, then `sanattaori/shishya-brain-life2`)

**Childhood, done 15:59 UTC** (code 0b0964d, 100M bytes, 6.2 h at 4,511 bytes/s): held-out **0.544** bits/byte.
That is 7.5% better than GPT (0.588); v1 Brain 0.561, BDH 0.532.

**Life test, done 17:10 UTC** (kernel `shishya-brain-life2`, code 09f8841; the output's `commit.txt` wrongly
shows v1's, copied from the mounted v1 run, fixed since): **FAIL**, but far ahead of v1 and of gradient learning.

#### Childhood (same bytes for every model)

| Model | Parameters | Bytes | Held-out bits/byte | Hours | Bytes/s |
|---|---:|---:|---:|---:|---:|
| gpt | 12,706,304 | 100,007,936 | 0.588 | 0.4 | 62,058 |
| bdh | 12,713,984 | 100,007,936 | 0.532 | 3.4 | 8,240 |
| brain | 12,795,916 | 100,007,936 | 0.544 | 6.2 | 4,511 |

Stage 1 (within 5% of GPT): BDH -9.5%, Brain -7.4%.

#### Life (test stream, fresh-context questions, chance = 20%)

| Learner | Setting (dev) | Recall, 3 exposures | 1 exposure | 1 correction (reward 3) | Changed fact: new wins | Paraphrased | Early facts kept | Bits/byte change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Shishya Brain** (local rule only) | 0.5 | 67% | 66% | 58% | 53% | 28% | 0.45 of 0.90 | -2.7% |
| GPT + fine-tuning | 3e-05 | 21% | 14% | 20% | 30% | 22% | 0.15 of 0.45 | +1.9% |
| BDH + fine-tuning | 3e-05 | 22% | 16% | 18% | 30% | 20% | 0.15 of 0.35 | +1.8% |
| No learning (as after childhood) | | brain 18% / gpt 17% / bdh 25% | | | | | | |

#### Verdict (rules fixed in docs/PLAN.md before the run)

| Rule | Result |
|---|---|
| L1_learns | **fail** |
| L2_no_forgetting | **fail** |
| L4_small_machine | **fail** |
| L3_beats_gradient | pass |

L3 detail: vs gpt: recall difference lower 95% bound +0.33, vs bdh: recall difference lower 95% bound +0.32
L4 measured: 529 ms per 256-byte write, peak 1247 MB.

**Overall: FAIL**


Memory diagnostic (`diagnose.json`, a made-up name two texts later, dev names): 3.52 bits/byte
with an empty memory → **0.08** with its memory (GPT 3.16, BDH 3.16).

The Brain's dev runs (the rule picked the rate with the highest dev recall):

| Write-rate multiplier | Dev recall, 3 exposures | Held-out bits/byte change |
|---:|---:|---:|
| 0.5 | 80% | -2.8% |
| 1.0 | 76% | -2.8% |
| 2.0 | 73% | -2.7% |
| 4.0 | 72% | -2.7% |

**What happened, in plain words**

- **v2 learns facts from reading alone, three times better than learning by gradients.** It recalls 67 of 100
  facts after 3 exposures, and 66% after a single one. GPT and BDH fine-tuned on every session: 21–22% and
  14–16%. Its reading got better (−2.7% bits/byte), not worse.
- **It misses the strict bar.** 80% after 3 exposures is needed; it got 67% (exactly 80% on the dev stream).
- **It still forgets over a long life.** The 20 earliest facts: 90% at the 30% mark, 45% at the end (v1: 80% → 5%).
- **Why, most likely: interference.**
  - Every step of every text is written (β ≈ rate × surprise).
  - Over 636 sessions that is about 160,000 writes into 16,384 units with 40 active each. Each unit is touched by
    hundreds of later writes.
  - The dev trend points the same way: the lower the write rate, the higher the recall (80%, 76%, 73%, 72%). The
    lowest rate tried was the best.
- **Questions in other words: 28%,** close to chance, as predicted when v2 was designed. Its addresses are the
  wording.
- **On a CPU:** 529 ms per session (Kaggle's CPU) at a 1.25 GB peak. Memory is within L4, speed is not.

**v1 → v2:** recall after 3 exposures 35% → 67%; after one exposure 24% → 66%; early facts kept 6% → 50%.

## 4. Why v2 forgets: the analysis behind v3 (dev stream only)

*Run on this CPU with the v2 model from Kaggle, 31 Oct 2026, with `python -m brain.study` (one life on the dev
stream, same seed as the life test's dev runs). The test stream was not looked at again: v3 is chosen on dev only.*

The CPU life gives the same dev recall as Kaggle's (80% at write rate 0.5), so the numbers below describe the run
that was judged.

**1. v2 learns every fact. Everything it misses, it forgot.**

- All 100 main facts were recalled right after their first reading, and right after each later one.
- The 50 facts read once: 100% right after reading, 56% at the end.
- Recall of all facts taught so far fell steadily as the life went on:

| Sessions lived | 106 | 212 | 318 | 424 | 530 | 636 |
|---|---:|---:|---:|---:|---:|---:|
| Facts taught so far | 65 | 112 | 161 | 202 | 227 | 260 |
| Recalled now | 95% | 89% | 79% | 73% | 71% | 67% |

**2. The older the fact, the more it is lost.** Main facts, by sessions since their last reading:

| Sessions since the 3rd reading | 0–100 | 100–200 | 200–400 | 400–700 |
|---|---:|---:|---:|---:|
| Recalled right after it | 100% | 100% | 100% | 100% |
| Recalled at the end | 100% (18) | 79% (33) | 92% (26) | 52% (23) |

**3. The cause: facts of the same kind share most of their address.**

v2's address is the last 32 bytes, each byte counting 0.9× the one after it. In "Pimev lives in a town called",
the name is 22 bytes back, so it counts about 0.1. The address is mostly the shared wording, and each new fact
of that kind overwrites part of the older ones. `python -m brain.probes window` (100 dev facts, untrained code):

| Kind of fact | How alike two facts' addresses are | Main facts recalled at the end |
|---|---:|---:|
| The secret word of … is | 18% | **96%** (28) |
| …'s pet is called | 44% | 71% (21) |
| …'s best friend is | 73% | 83% (12) |
| … has a toy named | 75% | 84% (19) |
| … lives in a town called | **86%** | 60% (20) |

The kind whose name sits next to the answer ("The secret word of Burof is") keeps almost everything. The kind
whose name is furthest from the answer keeps least.

**4. Most of what it writes is not facts.** Over the life, 162,180 steps were read. 66,678 of them were written
with strength above 0.01 and 24,763 above 0.5. Only 19% of all the writing went to the fact sentences; the rest
was ordinary story text.

**5. Other dev numbers** (one life at write rate 0.5): corrections (reward 3) 40% at the end, changed facts 63%
(the new name wins), questions in other words 21% (chance 20%), held-out bits/byte −3.1%, 471 ms per session on
this 4-core CPU.

**What it can do, shown** ([DEMO.md](DEMO.md), last section; dev names, v2 as trained): after reading "Naros's cat is called
Fugez." once inside a story and then 30 other stories, asked "Naros's cat is called", v2 *writes* "Fugez." (with
an empty memory: "Saram."). It takes a correction ("Danat.") and forgets on request (`forget`). After forgetting
the correction it writes "Sugez.": the first letter is gone, the rest of the old name is still stored under the
addresses that follow it.

## 5. Choosing v3 (dev stream only, v2's network)

Each idea was tried on the same dev life as section 4, with v2's trained network and only the memory's address or
write rule changed (`python -m brain.study … --set …`). The test stream was not used. Several lives shared the
4 CPU cores, so their "ms per session" are too high; speed is measured alone in the life test.

### The address (`python -m brain.probes`, 100 dev facts, untrained codes)

| Address | Question finds its own fact | Two facts of the same kind alike | Other kinds alike | A question in other words finds its fact |
|---|---:|---:|---:|---:|
| v2: last 32 bytes, fading 0.9 | 98% | 61% | 15% | 0% |
| Last 8 words, fading 0.9 | 100% | 32% | 7% | 5% |
| The words of the sentence (up to 8) | 100% | 30% | 7% | 7% |
| … each word counting by how new it is | 100% | **13%** | 3% | **41%** |

- **Words, not bytes:** a name four words back counts as much as the word just read.
- **The sentence:** the address starts after the last `.`, `!`, `?` or new line. A fact inside a story and the same
  words asked on their own get exactly the same address (own similarity 1.00 instead of 0.62). In the brain:
  event boundaries.
- **Novelty:** each earlier word counts by log((1+N)/(1+count)) / log(1+N). The counts come from the words of
  20,000 childhood stories (4.0M words), counted once and then fixed. A made-up name counts 1.0, "secret" 0.6,
  "called" 0.52, "is" 0.38, "the" 0.22. In the brain: the novelty signal.

### The dev lives

| Dev life (v2's network) | Recall, 3 readings | 1 reading | Correction | Changed fact: new wins | In other words | Early facts kept | Oldest facts kept (400+ sessions) | Bits/byte change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 (32 bytes) | 80% | 56% | 40% | 63% | 21% | 0.50 of 0.90 | 52% | -3.1% |
| Words of the sentence | 91% | 80% | 72% | 67% | 19% | 0.65 of 1.00 | 70% | -3.0% |
| … + write gate | 92% | 86% | 74% | 70% | 15% | 0.80 of 1.00 | 83% | -2.9% |
| … + metaplasticity | 100% | 96% | 92% | 3% | 21% | 1.00 of 1.00 | 100% | -3.0% |
| **… + novelty (v3)** | 99% | 98% | 100% | 83% | 28% | 0.95 of 1.00 | 96% | -2.8% |
| … + metaplasticity + reconsolidation | 96% | 74% | 92% | 90% | 23% | 0.95 of 1.00 | 96% | -2.9% |
| … + novelty + metaplasticity + reconsolidation | 99% | 100% | 100% | 93% | 27% | 0.95 of 1.00 | 96% | -2.8% |

- **Words** alone: recall 80% → 91%, and every kind of fact gains. The oldest facts still fade (70% kept).
- **The write gate** (only steps with surprise ≥ 2): a little better, 92%, early facts 0.80.
- **Metaplasticity** (ρ = 1 / (1 + use/τ), τ = 1, the local form of Larimar's least-squares write): nothing is
  forgotten (100%, early facts 1.00). But **a changed fact can no longer be learned: the new name wins 3% of the
  time.** The protected synapses refuse the correction.
- **Novelty:** 99% recall, 98% after one reading, 100% of corrections, the new name wins 83%, early facts 0.95,
  and questions in other words 28%. It passes L1 and L2 on dev with margin, and it can still change its mind.
- **Reconsolidation** (`meta_reopen`): a correction (reward > 1) writes with full plasticity, as a recalled
  memory becomes changeable again in the brain. It is meant to keep metaplasticity's protection without its
  refusal to update.

**The rule for the last choice, written before the reconsolidation lives finished (31 Oct, 21:58 UTC).**
v3 is words + sentence + novelty. Metaplasticity with reconsolidation is added only if, on the dev life, it keeps
more of the early facts than novelty alone (0.95), changed facts drop no more than 5 points below novelty's 83%,
and L1 and L2 still pass on dev. Otherwise v3 stays simpler.

**The result of that rule:** novelty + metaplasticity + reconsolidation kept the same share of the early facts
as novelty alone (0.95), so it is not added. v3 is **words of the sentence + novelty** (`models.V3`). The
combination was better at changed facts (93% against 83%) and after one reading (100% against 98%). It is a
candidate for v4, not part of v3.

Reconsolidation does what it was meant to: with it, metaplasticity's changed facts go from 3% to 90–93%.
