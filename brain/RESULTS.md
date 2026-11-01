# Shishya Brain: results

**Status, 1 Nov 2026, 10:40 UTC: four verdicts are in.** v1 and v2 FAIL on L1, L2, L4 (section 3). v3 passes L1–L3 and fails only L4, on CPU speed (section 6). v4 passes L1–L3 and fails the new rule L5, questions in held-out wordings: 35% against 50% (section 8; L4 is reported only from v4 on). Sections 4, 5 and 7 are the analysis behind v3 and v4 (dev stream only). Sections 1–2 are checks, not verdicts. The pass
rules are in [docs/PLAN.md](../docs/PLAN.md); each change was a dated amendment written before the run it applies to. Sections 1 and 2 are checks of the code and of the
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

## 6. The v3 verdict (Kaggle `sanattaori/shishya-brain-v3`, code b085a07, finished 23:20 UTC)

v3 is v2's trained network with v3's memory address (docs/PLAN.md, amendment of 31 Oct, committed in 543e43c
before this run). Same test stream, same GPT and BDH (their numbers are exactly those of the v1 and v2 runs), same
rules. The "brain" childhood row is v2's network.

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
| **Shishya Brain** (local rule only) | 0.5 | 99% | 96% | 90% | 77% | 34% | 0.95 of 1.00 | -2.7% |
| GPT + fine-tuning | 3e-05 | 21% | 14% | 20% | 30% | 22% | 0.15 of 0.45 | +1.9% |
| BDH + fine-tuning | 3e-05 | 22% | 16% | 18% | 30% | 20% | 0.15 of 0.35 | +1.8% |
| No learning (as after childhood) | | brain 18% / gpt 17% / bdh 25% | | | | | | |

#### Verdict (rules fixed in docs/PLAN.md before the run)

| Rule | Result |
|---|---|
| L1_learns | pass |
| L2_no_forgetting | pass |
| L4_small_machine | **fail** |
| L3_beats_gradient | pass |

L3 detail: vs gpt: recall difference lower 95% bound +0.70, vs bdh: recall difference lower 95% bound +0.69
L4 measured: 665 ms per 256-byte write, peak 1218 MB.

**Overall: FAIL**

L4 in detail:
- **Memory: within the rule** (1.2 GB of 2 GB).
- **Speed on Kaggle's CPU: fails.** 665 ms per session; the rule asks for 50 ms.
- On the GPU the same life took 38 ms per session.

**What happened, in plain words**

- **v3 passes the learning rules.**
  - It recalls 99 of the 100 main facts after 3 readings, and 96% after a single reading. It learns these from
    reading alone: no gradients, no optimiser.
  - The 20 earliest facts: 100% at the 30% mark, 95% at the end (L2 needs 80%).
  - Its reading got better, not worse: held-out bits/byte −2.7%.
- **It beats learning by gradients by about 4×.** GPT and BDH, fine-tuned on every session, recall 21–22%. The
  95% lower bound of the difference is +69 to +70 points (L3 needs > 0).
- **It still fails L4 on speed.** The BDH forward pass at this size takes about 50 billion operations per
  256-byte session (estimated from its sizes). The 50 ms target was too fast for it from the start; v1 and v2
  failed it the same way.
- **The test was not harder than dev this time:** dev 99%, test 99%. v2 had 80% and 67%.
- **Reported, not rules:**
  - Corrections (reward 3): 90%.
  - Changed facts: the new name wins 77%.
  - Questions in other words: 34%, against chance 20%. Still the weak spot.

**Against the predictions written before the run (docs/PLAN.md):** L1, L2 and L3 pass, as predicted. L4 fails on
speed, as predicted. Questions in other words came out a little above chance (34%), as predicted. The test was
not harder than dev, against the prediction that it might be.

| | v1 | v2 | **v3** |
|---|---:|---:|---:|
| Recall after 3 readings (L1: ≥ 80%) | 35% | 67% | **99%** |
| After 1 reading | 24% | 66% | **96%** |
| Correction (reward 3) | 24% | 58% | **90%** |
| Changed fact: new wins | 43% | 53% | **77%** |
| In other words | 25% | 28% | 34% |
| Early facts kept (L2: ≥ 0.80 of the 30% mark) | 0.05 of 0.80 | 0.45 of 0.90 | **0.95 of 1.00** |
| Held-out bits/byte change (L2: ≤ +5%) | −5.5% | −2.7% | −2.7% |
| CPU ms per session (L4: ≤ 50) | 562 | 529 | 665 |
| Verdict | FAIL (L1, L2, L4) | FAIL (L1, L2, L4) | **FAIL (L4 only)** |

## 7. Choosing v4 (dev stream only, v2's network)

v4 aims at questions in other words. The test stream gets wordings that were never used on dev
(`data.HELDOUT_PARAPHRASE`, committed in 2d8d9d7, before v4's address was designed).

### The meaning part of the address (`python -m brain.probes`, 100 dev facts, untrained codes)

v3's address plus a second part: the sentence's earlier words without their order, each counting by how new it
is, bound to the word being read (`key_bag` is its weight).

| Meaning part (key_bag) | Question finds its own fact | Same kind alike | In other words: finds its fact | In other words: share of the address |
|---:|---:|---:|---:|---:|
| 0 (v3) | 100% | 13% | 41% | 10% |
| 0.5 | 100% | 15% | 85% | 23% |
| 1 | 100% | 16% | 90% | 28% |
| 2 | 100% | 16% | 94% | 33% |
| 4 | 100% | 16% | 99% | 37% |
| 8 | 100% | 16% | 100% | 44% |
| 16 | 100% | 16% | 100% | 46% |

Two facts about the same person ("Zubak's cat is called", "Zubak lives in a town called") share 6% of their
address in v3, and 14–20% with the meaning part.

**The rule for choosing v4, written before any v4 dev life (1 Nov).**
1. The meaning part's weight is 4 or 8: whichever gives the higher dev recall in other words (a tie goes to 4).
   It must keep recall after 3 readings and the early facts kept within 2 points of v3's dev life (99%, 0.95).
   If neither does, v4 has no meaning part.
2. Then metaplasticity + reconsolidation (τ = 1) is added if, with the chosen weight, it raises dev changed facts
   by at least 5 points, keeps recall in other words within 2 points, and L1 and L2 still pass on dev.

### Two new baselines (dev stream, reported, not rules)

"Keep every text and look it up" is what people use today, so v4's life test reports it, in two forms. Both store
every sentence read (160 KB over the dev life). Before a question, they find the stored sentence whose words
match it best, with rare words counting most.

| Dev life | Recall, 3 readings | 1 reading | Correction | Changed fact | In other words |
|---|---:|---:|---:|---:|---:|
| GPT + retrieval (the sentence is put in front of the question) | 26% | 26% | 38% | 30% | 31% |
| A database: exact lookup, no neural reader | **100%** | **100%** | **100%** | **100%** | **100%** |

- **The lookup finds the right sentence every time** (20 of 20 checked by hand).
- **Our small GPT doesn't use it.** It (and BDH) doesn't copy a made-up name it has seen only once, even in the
  same text. After "Birab lives in a town called Vunev." it still pays 1.6–2.6 bits per letter for "Vunev" the
  second time. Copying common names ("Lily") works. Learning in context is weak at this size.
- **A plain database solves this test completely.** Shishya Brain shouldn't be sold as a better database.
  - Its claim is a *neural network* that learns from what it reads, after training, without gradients, with a
    memory of fixed size, and that keeps reading well.
  - Tests that a lookup can't pass on its own (two-step questions, using a fact in its own writing) are for
    later versions (docs/ROADMAP.md).

### The dev lives (v2's network)

| Dev life | Recall, 3 readings | 1 reading | Correction | Changed fact | In other words (dev wordings) | Early facts kept | Bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 (no meaning part) | 99% | 98% | 100% | 83% | **28%** | 0.95 of 1.00 | -2.8% |
| Meaning part, weight 4 | 98% | 90% | 84% | 60% | **49%** | 0.95 of 1.00 | -2.9% |
| Meaning part, weight 8 | 98% | 94% | 86% | 70% | **53%** | 0.90 of 1.00 | -2.8% |

**Step 1 of the rule:** weight 8 is higher in other words (53% against 49%), but its early facts kept (0.90)
are more than 2 points below v3's 0.95. Weight 4 meets both conditions (98%, 0.95). **v4's meaning part has
weight 4.**

**What the meaning part costs.** Reworded questions go from 28% to 49%. But the memory holds more overlap now
(other facts share 6% of an address instead of 3%), and some updating ability is lost: one reading 98% → 90%,
corrections 100% → 84%, changed facts 83% → 60%. Step 2 of the rule tests whether metaplasticity +
reconsolidation wins those back.

**Step 2 of the rule** (metaplasticity + reconsolidation, τ = 1, with weight 4):

| Dev life, v2's network | Recall, 3 readings | 1 reading | Correction | Changed fact | In other words (dev wordings) | Early facts kept | Bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3's memory | 99% | 98% | 100% | 83% | 28% | 0.95 of 1.00 | -2.8% |
| + meaning part (weight 4) | 98% | 90% | 84% | 60% | 49% | 0.95 of 1.00 | -2.9% |
| **+ metaplasticity + reconsolidation (v4)** | 100% | 94% | 98% | 83% | 47% | 1.00 of 1.00 | -2.9% |

- Changed facts rose by 23 points (the rule needs 5).
- Questions in other words stayed within 2 points (49% → 47%).
- L1 and L2 pass on dev (100%, early facts 1.00).

**So v4 is v3 + the meaning part (weight 4) + metaplasticity + reconsolidation** (`models.V4`). The amendment is
in docs/PLAN.md.

### What the v4 demo showed (dev names, v2's network with v4's memory)

- Reading and correcting work as in v3. One answer had a wrong letter ("Zigig." for Zifig), though the 1-in-5
  choice was still 100% right.
- **`forget`:** the correction is erased (1% left on "Danat"). But v4 then writes "Sugez.": a new first letter,
  then the rest of the *old* name.
  - The cause is the word code. The start of a word is the *sum* of its letters' codes (each bound to its
    place), so "Su…" and "Fu…" share half their address, and the old "…ugez" still answers.
  - The fix is for v5: code a word's letters in order (each bound to the code so far), not as a sum.
  - Also fixed on the way: `forget` now reopens protected synapses as a correction does (it didn't, with
    metaplasticity).
- **A question in other words** ("The town where Pimev lives is called") still failed in the demo ("Movis.";
  24% on Zifig). That fits the dev lives: v4 answers questions in other words about half the time (47%), not
  always.

## 8. The v4 verdict (Kaggle `sanattaori/shishya-brain-v4`, code 318800a, the amendment's own commit)

v4 is v2's network with v4's memory (docs/PLAN.md, amendment of 1 Nov). Same test stream, same GPT and BDH (their
numbers are again exactly those of the earlier runs), the rules as amended: L5 is new, L4 is reported only.

#### Childhood (same bytes for every model)

| Model | Parameters | Bytes | Held-out bits/byte | Hours | Bytes/s |
|---|---:|---:|---:|---:|---:|
| gpt | 12,706,304 | 100,007,936 | 0.588 | 0.4 | 62,058 |
| bdh | 12,713,984 | 100,007,936 | 0.532 | 3.4 | 8,240 |
| brain | 12,795,916 | 100,007,936 | 0.544 | 6.2 | 4,511 |

Stage 1 (within 5% of GPT): BDH -9.5%, Brain -7.4%.

#### Life (test stream, fresh-context questions, chance = 20%)

| Learner | Setting (dev) | Recall, 3 exposures | 1 exposure | 1 correction (reward 3) | Changed fact: new wins | Paraphrased | Held-out wording | Early facts kept | Bits/byte change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Shishya Brain** (local rule only) | 4.0 | 98% | 76% | 94% | 90% | 48% | 35% | 0.95 of 1.00 | -2.7% |
| GPT + fine-tuning | 3e-05 | 21% | 14% | 20% | 30% | 22% | 21% | 0.15 of 0.45 | +1.9% |
| BDH + fine-tuning | 3e-05 | 22% | 16% | 18% | 30% | 20% | 21% | 0.15 of 0.35 | +1.8% |
| GPT + retrieval (reported, not a rule) |  | 27% | 28% | 26% | 27% | 23% | 25% | 0.20 of 0.20 | +0.0% |
| A database: exact lookup, no reader (reported, not a rule) |  | 100% | 100% | 100% | 100% | 100% | 100% | 1.00 of 1.00 | +nan% |
| No learning (as after childhood) | | brain 18% / gpt 17% / bdh 25% | | | | | | | |

#### Verdict (rules fixed in docs/PLAN.md before the run)

| Rule | Result |
|---|---|
| L1_learns | pass |
| L2_no_forgetting | pass |
| L4_small_machine | **fail** (reported, not a pass rule) |
| L3_beats_gradient | pass |
| L5_other_words | **fail** |

L3 detail: vs gpt: recall difference lower 95% bound +0.69, vs bdh: recall difference lower 95% bound +0.68
L4 measured: 677 ms per 256-byte write, peak 1253 MB.

**Overall: FAIL**

**What happened, in plain words**

- **L5 fails, as predicted.**
  - Questions in the held-out wordings (never used on dev, written before v4 was designed) get **35%**. The bar
    was 50%, chance is 20%. The prediction written before the run was 35–50%, "more likely to fail".
  - Asked in the dev wordings, v4 gets **48%** on the test stream (v3 34%, v2 28%).
  - The meaning part helps, but it doesn't reach "understands questions in other words".
- **L1–L3 pass again.** 98% after 3 readings. Early facts kept 0.95. Reading −2.7%. About 4.5× the fine-tuned
  GPT and BDH (lower bounds +0.68 to +0.69).
- **Updating got better than v3.** Corrections 94% (v3 90%). Changed facts: the new name wins 90% (v3 77%).
  Metaplasticity with reconsolidation does what it was meant to.
- **After one reading: 76%,** below v3's 96%.
  - The dev choice of write rate picked 4.0: every rate gave 100% on dev, and 4.0 had the smallest bits-per-byte
    change by a hair.
  - On dev, rate 0.5 gave 94% after one reading. The higher rate likely costs the facts read once.
  - The dev rule doesn't look at one-reading recall; that is a weakness of the rule, to fix in v5's.
- **The two baselines.**
  - GPT + retrieval: 27%. Our small GPT doesn't copy a made-up name seen once.
  - The exact database lookup: 100% on everything, held-out wordings included, as predicted. Shishya is not a
    better database. Its claim is a neural network that learns after training, without gradients, at fixed size.
- **L4 (reported only):** 677 ms per session on Kaggle's CPU, 1.25 GB.

| | v3 | **v4** |
|---|---:|---:|
| Recall after 3 readings | 99% | 98% |
| After 1 reading | 96% | 76% |
| Correction | 90% | 94% |
| Changed fact: new wins | 77% | **90%** |
| In other words (the dev wordings) | 34% | **48%** |
| In other words (held-out wordings, L5 ≥ 50%) | not measured | 35% |
| Early facts kept | 0.95 | 0.95 |
| Verdict | FAIL (L4 only) | **FAIL (L5 only)** |
