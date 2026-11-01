# Shishya Brain: a small network that keeps learning after training

*Plan, written 31 Oct 2026, before any run. The pass rules below are fixed now and must not change after we see
results. The earlier "compile LLM tasks into programs" plan is parked in
[parked/compile-to-program.md](parked/compile-to-program.md).*

## The idea

LLMs stop learning when training ends. A brain never stops: every experience changes the strength of some
connections (synapses), and those strengths are the memory.

Shishya Brain is **born with backprop and lives with Hebbian learning**:

1. **Childhood:** an ordinary training run teaches it language. We use BDH (Pathway's "Dragon Hatchling",
   MIT licence), a network of neurons with sparse, positive activity and Hebbian working memory.
2. **Life:** after training, no more gradients. It keeps learning by itself: after reading any text, its
   **lasting synapses** change by a local rule, and the change stays across sessions.

## What we add to BDH

BDH already has fast synapses that live inside one text (its working memory). They are gone when the text ends.
We add **lasting synapses** to each layer: a matrix per head that is read on every step and written after every
text.

The write rule is local. Each synapse changes from what its own two neurons did, plus a third signal:

```text
Δ synapse(i → j) = rate × (activity of neuron i at step t)
                        × (error of neuron j at step t+1: what came next, minus what the synapse predicted)
                        × surprise(t) × reward
```

- **Pre before post (like spike-timing plasticity):** a context at step *t* is linked to what came at *t+1*.
  The memory learns "what follows what", not just "what was together".
- **Error-correcting (delta rule):** what is already known is barely written again. A changed fact overwrites
  the old one, instead of piling up next to it.
- **Surprise (like noradrenaline):** a byte the network didn't expect is stored more strongly. "Once upon a time"
  is stored weakly, a new name strongly.
- **Reward (like dopamine):** a correction from you comes with reward 3, and is stored 3 times as strongly
  (capped, so it stays stable).
- **Sparse:** keys are BDH's ReLU neurons, so a write touches only the synapses of the neurons that fired.

In childhood the read path and the write rate are learned by backprop through a few writes: it learns *how* to
learn. In life the rule runs on its own, with no gradients and no optimiser.

Not in this first test: sleep (moving lasting memory into the slow weights) and growing new neurons. They come
only if this test passes.

## The test

### Models, all the same size, all trained on the same bytes

| Model | Learns after training by |
|---|---|
| GPT (a plain transformer) | gradient fine-tuning on each session |
| BDH (Pathway's, unchanged) | gradient fine-tuning on each session |
| **Shishya Brain** (BDH + lasting synapses) | **its local rule only, no gradients** |

- **Data:** TinyStories (simple children's stories, CDLA-Sharing-1.0), read as bytes.
- **Childhood "memory drills":** 30% of training episodes put made-up facts in one text and ask about them two
  texts later. All three models see exactly the same episodes. Drill names come from a training list. The test
  uses different made-up names, never seen in childhood.
- **The gradient baselines get a fair chance:**
  - Their learning rate is chosen on a separate dev stream, from {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}: the highest dev
    recall among rates whose old-knowledge loss is ≤ 5%. If no rate meets that, the one with the least loss.
  - Their loss is weighted by the reward too.
  - The Brain gets the same kind of dev choice for one write-rate multiplier, from {0.5, 1, 2, 4}.

### The life stream

After childhood, each model lives through about 520 sessions of up to 256 bytes each:
- each session is an unseen story with up to 2 made-up facts in it
- half of the sessions are plain stories with no facts
- **Facts:**
  - 100 main facts, each seen 3 times, spread over the stream
  - 50 seen once
  - 50 seen once as a "correction" (reward 3)
  - 30 that change later ("Zib's cat is called Mork", then once "Zib's cat is called Pell")
- **Questions are asked in a fresh context**, with nothing before the question. Example: "Zib's cat is called"
  → which of 5 names fits best (the right one + 4 other names from the stream; chance = 20%).

### Pass rules (fixed now)

**Stage 1, childhood:** Brain and BDH reach a held-out bits per byte within 5% of GPT's, at the same bytes. This
is reported. If it fails, we still run the life test and say so.

**Stage 2, life:** the Brain passes only if all four hold.

| | Rule |
|---|---|
| L1 | **Learns:** at least 80% of the main facts recalled after 3 exposures |
| L2 | **Doesn't forget:** held-out bits per byte rises at most 5% over the whole stream, *and* the 20 earliest-learned facts keep at least 80% of the recall they had at the 30% mark |
| L3 | **Beats gradient learning:** against each gradient baseline (at its dev-chosen rate), either higher recall (95% paired-bootstrap lower bound > 0) with bits-per-byte rise no more than 1 point higher, or a bits-per-byte rise at least 2 points lower with recall no more than 5 points lower |
| L4 | **Small machine:** the life test runs on a CPU at peak memory ≤ 2 GB, ≤ 50 ms per write of a 256-byte session |

**Reported, not pass rules:**
- recall after 1 exposure
- reward vs no reward
- changed facts (does the new name win?)
- questions phrased in a way never seen in childhood

**If it fails,** we write down what failed and why, and decide with you what's next. No tuning on the test stream.

## Amendment, 31 Oct 2026 (before any v2 result): v2 addresses the memory differently

**Why.** The v1 Kaggle run had started. Meanwhile the CPU diagnostics on a tiny v1 Brain (15M bytes, dev names
only, never the test stream) showed:
- **Its memory barely helps recall a name two texts later:** 3.21 → 3.19 bits per byte.
- **The cause is the addresses.** At the step where a fact's name should be recalled, the addresses of different
  facts are 0.94–0.99 alike. A question finds its own fact 14–22% of the time, out of 100 facts.
- **Storage is not the problem:** inside one text it reads back fine.

**The change, v2 (`--model brain2`).**
- **Lasting memory addressed by recent input:** a fixed sparse code of the last 32 bytes. Each byte's random ±1
  code is multiplied by a random code for "how many steps ago", fading by 0.9 per step. The 40 strongest of
  16,384 units are kept (pattern separation, as in the hippocampus).
- **Not trained:** like the random projections that index memories in the dentate gyrus.
- **Everything else is the same as v1:** the BDH core, the values, the read path, the surprise- and
  reward-weighted delta rule, the training recipe, the life stream.
- **Evidence for it:** on the same 100 dev facts, untrained, a question finds its own fact 96% of the time. The
  window and fading were chosen there, on dev names only.

**Rules.**
- v2 is judged by **the same pass rules**, on the same test stream, against the same GPT and BDH runs.
- v1's verdict is reported as it comes, whatever v2 does.
- Prediction, written now: questions phrased differently (reported, not a rule) will do worse in v2. The address
  is the wording.

## Amendment, 31 Oct 2026 (after v2's verdict, before any v3 run): v3 addresses the memory by the sentence's words, the least familiar counting most

**Why.** v2 failed L1 (67% after 3 exposures), L2 (the 20 earliest facts kept 0.45 of their 0.90) and L4 (speed).
The analysis behind v3 used the dev stream only. The test stream was not looked at again; see
[brain/RESULTS.md](../brain/RESULTS.md) section 4.
- **v2 learns every fact and then loses some.** All 100 main facts were recalled right after each reading. At the
  end, 80% were (dev).
- **The cause is the address.**
  - v2 addresses its memory by the last 32 bytes, each counting 0.9× the one after it. In "Pimev lives in a town
    called", the name counts about 0.1.
  - Two facts of the same kind have addresses 61% alike (86% for "… lives in a town called"), so each new fact
    overwrites part of the older ones.
  - The kind with the least overlap kept 96%; the most overlapped, 60%.

**The change, v3 (`--model brain3`, settings in `models.V3`).**

1. **Address by words, not bytes.** The address is the last 8 words, each word's code built from its bytes, and
   every word counts the same. A name four words back counts as much as the word just read.
2. **The address starts at the sentence.** Words before the last `.`, `!`, `?` or new line are left out (in the
   brain: event boundaries). A fact inside a story and the same words asked on their own get the same address.
3. **Each word counts by how new it is.**
   - The weight is log((1+N)/(1+count)) / log(1+N), from the words of the first 20,000 childhood stories (4.0M
     words), counted once and then fixed (in the brain: the novelty signal).
   - A made-up name counts 1.0; "called" 0.52; "the" 0.22.
   - Two facts of the same kind then share 13% of their address, not 61%.
4. **The same memory as v2.** Same 16,384 units, 40 active; same delta rule, values and read path.

- **Everything else is v2's.** The network is v2's, as trained in childhood (no new childhood): the BDH core, the
  read path, the values, the learned write rates. So are the delta rule, surprise and reward, the life stream,
  the dev choice of the write-rate multiplier from {0.5, 1, 2, 4}, and the GPT and BDH runs it is compared with.
  `brain/make_v3.py` builds it from the v2 run.
- **`forget` is new, not a rule.** It erases a text on request (Larimar's forgetting, done with the local rule)
  and is shown in [brain/DEMO.md](../brain/DEMO.md).

**Evidence (dev stream only, v2's network, one life each, `python -m brain.study`).**

| Dev life, v2's network | Recall, 3 readings | 1 reading | Correction | Changed fact | In other words | Early facts kept | Bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2's memory | 80% | 56% | 40% | 63% | 21% | 0.50 of 0.90 | −3.1% |
| **v3's memory** | **99%** | **98%** | **100%** | **83%** | 28% | **0.95 of 1.00** | −2.8% |

Six other lives were tried, with a write gate, metaplasticity (Larimar's protection, done locally) and
reconsolidation. They are all in brain/RESULTS.md section 5, with the rule for the last choice, which was
committed before its result (1cf73cf).

**Rules.**
- v3 is judged by the same pass rules, on the same test stream, against the same GPT and BDH. L4 stays as written.
- v1's and v2's verdicts stand.
- Stage 1 (childhood) is v2's: held-out 0.544 bits/byte.

**Predictions, written now:**
- L1, L2 and L3 pass. The test stream has been harder than dev: v2 got 80% on dev and 67% on test. So L1 and
  L2's early-facts part may land nearer their bars than on dev.
- L4 fails on speed (about 0.5 s per session; the rule asks for 50 ms).
- Questions in other words: a little above chance (dev 28%), far below Larimar's 83–88%.

**Decision point (agreed with the owner, 31 Oct).**
- If v3 passes L1 and L2, the next steps are a meaning-based address for questions in other words, and sleep.
- If it doesn't, v1–v3 are written up as a finished study.

## Amendment, 1 Nov 2026 (after v3's verdict, before any v4 run): v4 adds a meaning part to the address, and rule L5

**The owner's decisions (1 Nov):**
- Add intelligence and test it.
- Set speed aside for now: **L4 is reported, not a pass rule, from v4 on.** v1–v3's verdicts stand as they were.

**Why.** v3 passed L1–L3. Its weak spot is questions in other words: 34% on the test, chance 20%. Its address is
the words in their order, so "The town where Pimev lives is called" doesn't find "Pimev lives in a town called".

**The change, v4 (`--model brain4`, settings in `models.V4`).**
1. **v3's address, plus a meaning part.**
   - The meaning part is the sentence's earlier words without their order, each counting by how new it is,
     bound to the word being read.
   - Its weight is 4 against the ordered part's 1.
   - The same words in another order give the same meaning part. So a question in other words shares about 37%
     of its fact's address instead of 10%, and finds its own fact among 100 dev facts 99% of the time instead
     of 41%.
2. **Metaplasticity with reconsolidation.**
   - A unit that has stored a lot changes less: plasticity 1 / (1 + use/τ), τ = 1. This is the local form of
     Larimar's least-squares write.
   - A correction (reward > 1) writes at full plasticity again, as a recalled memory becomes changeable in the
     brain.
   - Added by step 2 of the rule written before the v4 dev lives (brain/RESULTS.md section 7). With the meaning
     part, it raised changed facts from 60% to 83% and kept questions in other words within 2 points.
3. **Everything else is v3's.** v2's network (no new childhood), the delta rule, the life stream, the dev choice
   of the write-rate multiplier, and the GPT and BDH runs.

**Evidence (dev stream only, v2's network, one life each; brain/RESULTS.md section 7).**

| Dev life, v2's network | Recall, 3 readings | 1 reading | Correction | Changed fact | In other words (dev wordings) | Early facts kept | Bits/byte |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3's memory | 99% | 98% | 100% | 83% | 28% | 0.95 of 1.00 | -2.8% |
| + meaning part (weight 4) | 98% | 90% | 84% | 60% | 49% | 0.95 of 1.00 | -2.9% |
| **+ metaplasticity + reconsolidation (v4)** | 100% | 94% | 98% | 83% | 47% | 1.00 of 1.00 | -2.9% |

**A new test: held-out wordings.**
- A second way to ask each kind of question (`data.HELDOUT_PARAPHRASE`, for example "The word that Zubak keeps
  secret is").
- They were committed (2d8d9d7) before v4's address was designed, and are used only on the test stream. So v4
  can't have been fitted to them.

**Rules.** L1, L2 and L3 as before. L4 is reported only. And a new one:

| | Rule |
|---|---|
| L5 | **Understands questions in other words:** at least 50% of the main facts recalled when asked in the held-out wordings (chance 20%) |

The verdict passes if L1, L2, L3 and L5 all hold.

**Reported, not rules:**
- Two baselines: GPT + retrieval, and an exact database lookup (dev: 26% and 100%).
- The dev wordings, and everything reported before.

**Predictions, written now:**
- L1, L2 and L3 pass.
- L5 is uncertain, more likely to fail than pass. The dev wordings give 47%, and the held-out wordings change
  more words ("whose name is", "likes best", "keeps secret"). I expect 35–50%.
- The database lookup scores close to 100% on the test too. Shishya doesn't beat a database at storing facts;
  that's not its claim.
- L4 (reported): about 0.7 s per session on Kaggle's CPU.

## Sizes and compute

| Preset | Where | Brain | GPT | Childhood |
|---|---|---|---|---|
| `tiny` | this CPU, to check the code works | 4 layers, 128 wide, 512 neurons/head (≈0.9M) | ≈0.9M | a few million bytes, minutes |
| `bee` | Kaggle's free T4 | 6 layers, 256 wide, 4,096 neurons/head (≈12.9M) | ≈12.7M | 100M bytes each, a few GPU hours |

The Brain's lasting synapses at `bee` size are 6 × 4 × 4,096 × 64 numbers, about 25 MB.

## Steps

1. Code: models, data, childhood, life test, tests (including that our BDH matches Pathway's exactly).
2. `tiny` run on this CPU: the code works end to end. Its numbers are not the verdict.
3. `bee` run on Kaggle: childhood for the 3 models, then the dev stream, then the test stream. Verdict by the
   rules above, in `brain/RESULTS.md`.
4. Only if it passes: sleep, longer lives, learning from your corrections in a real task.

## Prior work this builds on

- BDH: Kosowski et al., *The Dragon Hatchling* (Pathway, arXiv 2509.26507, 2025). Code: github.com/pathwaycom/bdh
  (MIT), vendored unchanged in `brain/third_party/`.
- Delta rule in parallel form: Yang et al., *Parallelizing Linear Transformers with the Delta Rule* (2024).
- Multi-speed memory: Behrouz et al., *Nested Learning / HOPE* (Google, NeurIPS 2025); *Titans* (2025).
- Sleep consolidation: Lee et al., *Language Models Need Sleep* (arXiv 2605.26099, 2026).
- Learning facts without forgetting: Lin et al., *Continual Learning via Sparse Memory Finetuning* (Meta, 2025).
- One-shot episodic memory for facts: Das et al., *Larimar: Large Language Models with Episodic Memory Control*
  (IBM, ICML 2024; github.com/IBM/larimar, Apache-2.0). A BERT encoder, a memory matrix written by least squares,
  and a GPT-2/GPT-J decoder. It edits facts in one shot and can forget on request. Unlike us, it is an add-on to an
  LLM, learns only through explicit edits, and its write is global, not local.
- TinyStories: Eldan and Li (2023).
