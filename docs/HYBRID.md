# Shishya hybrid: a frozen LLM with Shishya's lasting memory (plan and pass rules)

*Written 1 Nov 2026 on branch `brain-hybrid`, before any full dev or test life. The rules and predictions below
don't change after results are seen. Code: `brain/hybrid.py`.*

## The question

v3 showed that Shishya's lasting memory learns facts after training, from one reading, without gradients
(brain/RESULTS.md section 6). But its core is a 12.8M network that only writes children's stories. This test asks:

> **Does the same memory work on a real, frozen LLM, and how does it compare with RAG on that same LLM?**

If it does, the claim becomes "Shishya makes any LLM keep learning after it ships". The LLM supplies the language
and knowledge; the memory supplies learning after training.

## The design (stage 1, no training at all)

- **The LLM:** SmolLM2-135M (Apache-2.0), frozen. Its weights never change. 135M rather than 360M because this
  CPU (4 ARM cores, no GPU) runs 360M about 2.6× slower. 360M or larger is for a later GPU run.
- **The memory:** v3's, unchanged. The fixed sparse address (the words of the sentence, each counting by how new
  it is; 16,384 units, 40 active), optionally with v4's meaning part (`key_bag`). The same exact delta rule,
  β = min(rate × surprise × reward, 1). Size fixed: 16,384 × 576 numbers (38 MB).
- **What is stored:** at each token, the address of the text so far is linked to the next token's unit-length
  embedding (the LLM's own).
- **How it is read:** the address reads back a vector; its likeness to each token's embedding, times `gain`, is
  added to the LLM's next-token scores.
- **Surprise:** the LLM's loss on the token divided by its average loss on held-out stories, capped at 4.
- **Nothing is trained.** `rate`, `gain` and `key_bag` are chosen on the dev stream.

The smoke check (a 12-fact dev life, not a result): memory 100% recall, RAG 100%, both 100% on changed facts;
in other words memory 17%, RAG 100%.

## The test

The same life test as v1–v4 (`brain/lifetest.py`): the same dev and test streams (seeds 11 and 22), 100 main facts
read 3 times, 50 read once, 50 corrections (reward 3), 30 changed facts, about 636 sessions. Questions are asked in
a fresh context; a fact counts when the right name is the most likely of 5. The test stream gets the held-out
wordings (`data.HELDOUT_PARAPHRASE`), never used on dev. All learners use the same frozen LLM:

| Learner | What it does |
|---|---|
| **Hybrid** | the LLM + Shishya's memory (local rule only) |
| RAG | keep every sentence read; put the best match (rare words count most) before the question |
| Fine-tuning | a copy of the LLM; one Adam step per session (loss × reward) |
| No learning | the LLM as it is |
| Lookup | a database, no reader: the ceiling for this test (reported only) |

**The dev choice (fixed now).**
- Hybrid grid: `key_bag` ∈ {0, 4}, `rate` ∈ {0.25, 0.5, 1}, `gain` ∈ {2, 4, 8} (18 lives). Among settings whose
  held-out bits per byte rise ≤ 5%, the one with the highest dev recall (3 readings) + dev recall in other words;
  a tie goes to the smaller rise.
- Fine-tuning: learning rate ∈ {3e-6, 1e-5, 3e-5}, chosen by the rule used since v1 (`lifetest.choose`): the
  highest dev recall whose bits per byte rise ≤ 5%.
- The test stream is run once, with the chosen settings.

## Pass rules (test stream)

| Rule | Passes when |
|---|---|
| H1 learns | hybrid recall after 3 readings ≥ 80% |
| H2 no forgetting | early facts kept ≥ 0.80 of their level at the 30% mark, and held-out bits per byte rise ≤ 5% |
| H3 beats fine-tuning | 95% lower bound of (hybrid − fine-tuning) recall > 0 (paired bootstrap over facts) |
| H4 matches RAG | 95% lower bound of (hybrid − RAG) recall > −5 points |
| H5 other words | hybrid recall with the held-out wordings ≥ 60% |

Overall pass: all five. Reported, not rules: one reading, corrections, changed facts, memory size against RAG's
stored text, ms per session on this CPU.

## Predictions (written before the runs)

- H1, H2, H3 pass. H4 passes: both near 100% on facts asked in the same words.
- **H5 fails** (about 20–50%). The memory's address is still mostly the wording, while RAG matches the rare words
  of any wording and the LLM reads the sentence. Overall: **FAIL on H5**.
- RAG beats the hybrid clearly on questions in other words. The hybrid's edge is a fixed-size memory with no text
  kept, and held-out reading that gets better, not worse.

## What happens next (fixed now)

- **If H1–H4 pass:** stage 2, on a GPU. A larger LLM (360M or more); a read path into the LLM's hidden states,
  trained once and then frozen, so the LLM can *use* a fact, not only repeat it; and two-step questions ("where
  does Zubak's best friend live?"), against RAG.
  **The stop rule:** if stage 2's hybrid doesn't beat RAG on two-step questions (95% lower bound > 0), the memory
  stops as a research line and becomes a product feature (learning each correction on the next request).
- **If H1 or H2 fails:** v3's memory doesn't carry over to a real LLM as built. Write it up, and look at why before
  any stage 2.
- **H5** decides whether v4's meaning address is the way forward or whether the address needs the LLM's own
  hidden states (stage 2).

## Results (1 Nov 2026, this CPU; `brain/out/hybrid/`, not in git)

**Verdict: FAIL on H4 and H5.** H1, H2 and H3 pass.

### Dev (the choice)

The rule chose `key_bag` 0, `rate` 0.25, `gain` 8: dev recall 99%, in other words 17%, bits per byte +4.4%. The
best setting with the meaning part (`key_bag` 4, rate 0.25, gain 8) had recall 88% and other words 44%, but bits per
byte +5.1%, just over the limit, so it was out. Gain 2 was too weak to beat the LLM's own guesses (recall 41–48%).
Gain 8 starts to push stored words into ordinary text (bits per byte +4–17%). Fine-tuning: 3e-5 chosen (dev recall
25%).

### Test (run once)

| Learner (SmolLM2-135M, frozen) | Recall, 3 readings | 1 reading | Correction | Changed fact | Other words (dev wording) | Other words (held-out) | Early facts kept | Bits/byte | Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid** | 96% | 76% | 88% | 97% | 30% | 27% | 0.95 | +2.5% | 38 MB (fixed) |
| RAG | 100% | 100% | 100% | 100% | 100% | 100% | 1.00 | 0 | 160 KB of text |
| Lookup (a database) | 100% | 100% | 100% | 100% | 100% | 100% | 1.00 | | 160 KB |
| Fine-tuning (3e-5) | 36% | 6% | 34% | 20% | 29% | 27% | 0.71 | −25.8% | |
| No learning | 23% | 16% | 10% | 13% | 24% | 19% | | 0 | |

| Rule | Result |
|---|---|
| H1 learns (≥ 80%) | pass (96%) |
| H2 no forgetting | pass (0.95 kept; +2.5%) |
| H3 beats fine-tuning | pass (lower bound +50 points) |
| H4 matches RAG (lower bound > −5 points) | **fail** (−8 points; 96% against 100%) |
| H5 other words (≥ 60%) | **fail** (27%; chance 20%) |

Hybrid: 348 ms per session on this CPU (the LLM's forward pass is most of it).

### Against the predictions

- H1, H2, H3 passed, as predicted. H5 failed, as predicted (27%, inside the 20–50% guess).
- **H4 was predicted to pass and failed.** RAG got every fact; the hybrid missed 4 of 100.
- **The memory-size argument doesn't hold at this scale.** The hybrid's fixed memory is 38 MB. RAG kept 160 KB of
  text, 236× less. A fixed size only pays off after hundreds of MB of text read.

### What it means, in plain words

- **v3's memory carries over to a real LLM without any training:** 96% of facts after 3 readings, 97% of changed
  facts, its early facts kept, and the LLM's reading barely touched. That is 2.7× fine-tuning (36%).
- **On this test, RAG on the same LLM is better at everything:** facts, other wordings, memory size and speed.
  Stage 1 has no advantage over RAG.
- The rule above sent stage 2 on only if H1–H4 passed. H4 failed, so stage 2 does not start by this plan. The
  plan didn't say what to do when only H4 fails; that is the owner's decision.
