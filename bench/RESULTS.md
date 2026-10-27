# rulebench v0: results

**Setup (27 Oct 2026):**
- **Machine:** Apple M2, 16 GB.
- **Decision model:** [Kev-0.8B](https://github.com/jaredpalmer/kev) (Apache-2.0), served locally with MLX (bf16), through `SystemOneBackend`.
- **LLM baseline:** `nvidia/nemotron-3.5-super-vl:free` on OpenRouter.
- **Tasks:** 60 in total, 20 each of leave requests, stock alerts and shell commands (`bench/make_tasks.py`, seed 7).
- **Raw answers:** `bench/out/*.json` (not committed; rerun to reproduce).

**What is measured:** whether the main field (verdict or action) matches the correct answer, and whether the answer breaks any of the scenario's rules. The model is never shown the rules. The facts are shown to every system.

> The Shishya authors wrote this benchmark, so it favours the cases rules are good at. Read it as a demonstration of the mechanism, not as a measure of model quality.

## 1. The same model call, with and without Shishya

| System | Correct | Breaks a rule | Answered | p50 | p95 | LLM calls/task |
|---|---:|---:|---:|---:|---:|---:|
| Kev-0.8B alone | 42% | **38%** | 100% | 184 ms | 250 ms | 0 |
| **Kev-0.8B + Shishya** | **73%** | **0%** | 100% | **195 ms** | **252 ms** | 0 |
| LLM writes JSON, retries up to 3× when a rule fails | 83% | 0% | 95% | 5,048 ms | 42,685 ms | 1.2 |

By scenario (correct / breaks a rule, out of 20):

| Scenario | Kev alone | Kev + Shishya | LLM + retry |
|---|---|---|---|
| Leave requests | 11 / 6 | 17 / 0 | 16 / 0 |
| Stock alerts | 8 / 3 | 11 / 0 | 19 / 0 |
| Shell commands | 6 / 14 | 16 / 0 | 15 / 0 |

**What happened:**
- **Kev alone broke a rule on 23 of 60 tasks:**
  - it said `run` for 14 shell commands, including `rm -rf / --no-preserve-root` and `mkfs.ext4 /dev/sda1`
  - it approved 6 leave requests beyond the employee's balance or probation limit
  - it said `add` (buy more) 3 times while the market guard was on
- **Shishya replaced every one of those** with the most likely allowed answer, in about 11 ms more than the model call itself, with no extra model calls. 19 of the 23 replacements were exactly the correct answer. The other 4 were catastrophic shell commands, where it returned `ask` and the ideal answer was `block`: safe, but not ideal. The model gave `block` too little probability.
- **The LLM judged the text better**, especially stock news (19/20 vs 11/20). Shishya can't make a 0.8B model understand an auditor's resignation; it can only stop answers the rules forbid. The LLM was also 26× slower at the median and up to 43 s at p95, broke a rule on its first try 8 times, and gave no valid answer on 3 tasks.

## 2. Shishya with a Guru

Kev-0.8B + Shishya, asking the same LLM only when the best allowed answer's probability is below 0.5. The threshold was fixed before the run.

| System | Correct | Breaks a rule | p50 | p95 | Guru used |
|---|---:|---:|---:|---:|---:|
| Kev-0.8B + Shishya | 73% | 0% | 195 ms | 252 ms | 0 / 60 |
| Kev-0.8B + Shishya + Guru, bar on the *whole answer* (p < 0.5) | 77% | 0% | 5,716 ms | 20,575 ms | **58 / 60** |
| Kev-0.8B + Shishya + Guru, bar *per field* (any field < 0.5) | 73% | 0% | 3,411 ms | 17,173 ms | 42 / 60 |
| LLM writes JSON + retry ≤3 | 83% | 0% | 5,048 ms | 42,685 ms | (always) |

**Lesson (a design flaw this run exposed):** in v0.1, `min_probability` is compared with the *joint* probability of the whole answer, the product over all fields. With two or three fields and a small model, that product is almost always below 0.5, so 58 of 60 tasks went to the slow Guru. The two answers that stayed on the fast path were both right.

**After the fix (v0.1.0 ships with it):** the bar is now compared per field, through `Decision.confidence`, the weakest field's probability. The Guru was asked on 42 of 60 tasks instead of 58, but overall accuracy didn't improve. The 18 answers that stayed on the fast path were 78% right, and the Guru (free Nemotron, given the broken rules) was 71% right on the 42 hard ones. Kev-0.8B's confidence doesn't yet separate easy cases from hard ones well, and on these hard cases a free LLM isn't much better than the small model. Better calibration and a stronger specialist come before the Guru pays off.

## Takeaways

1. **The rule search works as designed:** zero rule breaks, exact best allowed answer, ~11 ms on top of the model call.
2. **A 0.8B decision model is weak at judging financial news.** That's the next target: a small specialist trained on free GPUs, labelled by two open-weight teachers that must agree, and tested on human-labelled data it never saw.
3. **The Guru is only worth it when the small model's confidence means something.** Per-field bars shipped in v0.1.0; calibration on your own labelled data comes next.
