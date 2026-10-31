# Shishya Brain: results

**Status, 31 Oct 2026, 14:30 UTC: the verdict runs are still on Kaggle. Nothing below is the verdict.** The pass
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

**A measurement bug, fixed (b0e2b6c and later).** The CPU check ran in a child process and read its peak memory
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

| Run | Kaggle kernel | State |
|---|---|---|
| v1: GPT, BDH, Brain v1 (100M bytes each), then the life test | `sanattaori/shishya-brain-life` | running since 07:32 UTC |
| v2: Brain v2 (100M bytes) | `sanattaori/shishya-brain-v2` | running since 09:48 UTC |
| v2 life test, against v1's GPT and BDH | `sanattaori/shishya-brain-life2` | after both |

Tables will come from `python -m brain.report <run>/out`.
