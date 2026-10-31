# Shishya Brain: results

**Status, 31 Oct 2026, 14:20 UTC: the verdict runs are still on Kaggle. Nothing below is the verdict.** The pass
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

### Tiny v2, 15M bytes

**Pending.** Held-out bits/byte at step 800 of 916: 1.004 (v1 at the same step: 1.106).

## 2. Kaggle canary (2× T4, 20 steps per model, full `bee` size)

| | GPT | BDH | Brain v1 |
|---|---:|---:|---:|
| Training speed (bytes/s) | 45,176 | 6,864 | 3,162 |
| Parameters | 12,706,304 | 12,713,984 | 12,845,068 |

On a CPU (Kaggle's), the `bee` Brain v1 takes **625 ms** per 256-byte session. Rule L4 says ≤ 50 ms, so L4 fails as
written. The 5.5 GB peak memory measured there was mostly the evaluation scoring 256 questions at once. That was
fixed in 0484c13; v1's memory is re-measured with the fixed code after its run.

## 3. The verdict runs

| Run | Kaggle kernel | State |
|---|---|---|
| v1: GPT, BDH, Brain v1 (100M bytes each), then the life test | `sanattaori/shishya-brain-life` | running since 07:32 UTC |
| v2: Brain v2 (100M bytes) | `sanattaori/shishya-brain-v2` | running since 09:48 UTC |
| v2 life test, against v1's GPT and BDH | `sanattaori/shishya-brain-life2` | after both |

Tables will come from `python -m brain.report <run>/out`.
