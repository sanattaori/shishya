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
