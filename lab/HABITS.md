# Can a correction change how an LLM behaves, with no written instruction? (plan, before any run)

*Written 3 Nov 2026, 03:35 IST, before any result. Rules and predictions don't change after results. Nothing here
touches BDH or the work on `brain-v5`.*

## Why

Today, telling an LLM "keep it short" means repeating that sentence in every prompt. People form habits: at first you
remember the rule, later you just do it. The idea here: a correction starts as a note. At night the model practises
with the note and without it, and the difference in its own activity becomes a small **habit vector**. Then the note
can be dropped, and the vector is added inside the model while it answers. No gradients, no training, no prompt text,
and a few KB per habit. It can be undone by removing it.

Prior work has the mechanism: steering vectors, task vectors, persona vectors. I found no test of it as habit
forming in a personal AI that grows from its own corrections. That search was limited.

## The test

- **Model:** SmolLM2-1.7B-Instruct (Apache-2.0), local, on a CPU. (The plain 1.7B has no chat format; changed before any
  run.) Fixed system prompt: "You are a helpful assistant."
- **Habits, each from one instruction:**
  1. Short: "Always answer in one short sentence."
  2. Asks back: "After answering, always ask the person one question back."
  3. Admits: "If you do not know something for certain, say you don't know."
  4. Name: "Your name is Shishya. You are an AI student."
- **Questions:** practice (only to build vectors), dev (only to choose settings), test (used once, at the end). They
  don't overlap.
- **Settings chosen on dev only:** the layer, the strength, and how the vector is built: from the last prompt token, or
  from replies written with and without the note. A habit's dev score is its pass measure, among settings that keep
  the guards.
- **Compared on test:** no help; the usual way (the instruction in the prompt); the habit vector, no text.

## Pass rules (test questions)

| Habit | Passes when |
|---|---|
| Short | mean words at most 50% of no help |
| Asks back | replies with a question at least 30 points above no help |
| Admits | admits on made-up questions at least 30 points above no help |
| Name | says "Shishya" in at least 60% of identity questions |
| Guards, for each | easy facts right at most 10 points below no help; at most 5% of replies are garbage |

**Overall pass: Short and at least one other, each with its guards.** Reported, not rules: the usual way next to it;
extra tokens; time per reply; vector size.

## Predictions (before any run)

Short passes 60%. Asks back 35%. Admits 30%. Name 15%. Overall pass 45%. I expect the usual way to beat the vector on
most habits, and the vector to win only on prompt tokens. The name is the hardest: a vector leans toward a way of
behaving, and a rare word is not a way of behaving.

## What happens next (fixed now)

- **Overall pass:** sleep in `mind/` turns corrections into habit vectors, and two copies with different lives are
  compared on the same questions.
- **Overall fail:** try a learned version: the same few thousand numbers, fitted by gradients for a few minutes at
  night. If that fails too, habits stay as text, and this idea is written up as tried.

## Results (3 Nov 2026, 05:27 IST; test questions, once)

**The verdict: FAIL.** No habit passed. The overall rule needed Short and one other.

| Habit | No help | The usual way (a sentence in the prompt) | Habit vector, no text | Needed | |
|---|---:|---:|---:|---|---|
| Short: mean words | 63 | 21 | **13** | at most 31.5 | bar met, **guard fails**: easy facts 70% against 100% |
| Asks back: replies with a question | 0% | 33% | no setting kept the guards on dev | at least 30% | fail |
| Admits: on made-up questions | 35% | **65%** | 50% | at least 65% | fail (facts stayed 100%) |
| Name: says "Shishya" | 0% | **100%** | no setting worked on dev | at least 60% | fail |

- The usual way kept easy facts at 100% on every habit, for a cost of only 7 to 15 prompt tokens.
- The Short vector is the one striking number, but it gets there by making things up ("245 degrees Fahrenheit").
- A strong Admits vector refused everything on dev: 100% on made-up questions, 0% on easy facts. The guard caught it.
- A vector of about 8 KB is not worth much when the sentence it replaces costs 7 to 15 tokens and works better.

**Against the predictions.** I put Short at 60%, Asks at 35%, Admits at 30%, Name at 15% and overall at 45%. I was right
that the usual way would win on most habits, and that the name would be the hardest.

**What follows (fixed above):** the learned version. See `lab/SELF_LAYER.md`.
