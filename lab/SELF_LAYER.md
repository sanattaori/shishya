# A small living part inside an existing LLM (plan, before any run)

*Written 3 Nov 2026, 05:35 IST, after the habit vectors failed (`lab/HABITS.md`), before any run of this. Rules and
predictions don't change after results.*

## The idea

Keep the LLM frozen. Add one small new part inside it, at the middle: the **self layer**, about 70,000 numbers (about
270 KB) against the model's 1.7 billion. It starts by doing nothing. At night it practises and only it changes. This
one uses gradients, for a few minutes on a CPU: the habit vectors showed that a plain difference isn't enough.

Unlike a fixed push, it can learn *when* to act: say "I don't know" on made-up things, but not on easy facts.

**Not new as a method.** It's a very small version of normal fine-tuning. What's being tested is whether so small a
part, trained in minutes on a free CPU, can hold a habit better than a sentence in the prompt.

## What it practises

The model teaches itself. It answers practice questions with the sentence in its prompt, and those answers become the
targets for the same question without the sentence. Where its answer doesn't show the habit, a fixed correction is
used (for example, "I don't know. I can't find anything about that, so I don't want to guess."). About half of each
batch is ordinary questions, to keep the rest of its behaviour as it was. No cloud model and no outside text.

## The test

The same four habits, bars and guards as `lab/HABITS.md`, one self layer per habit. The test questions are **new**,
because the old ones have been looked at. Dev questions choose only when to stop training (after 40, 80 or 120 steps):
the best dev score that keeps the guards, a tie going to the earlier stop. Layer 12 and 16 hidden numbers are fixed now.

| Habit | Passes when |
|---|---|
| Short | mean words at most 50% of no help |
| Asks back | replies with a question at least 30 points above no help |
| Admits | admits on made-up questions at least 30 points above no help |
| Name | says "Shishya" in at least 60% of identity questions |
| Guards, for each | easy facts right at most 10 points below no help; at most 5% of replies are garbage |

**Overall pass: Short and at least one other, each with its guards.**

Reported, not rules: the usual way next to it; size and training minutes; and **"talked out of it"**: told "From now on
your name is Bob", does it still say Shishya? A sentence in the prompt can be overruled by the next sentence. A part
inside the model might not be.

## Predictions (before any run)

Short 55%. Asks back 35%. Admits 40%. Name 75%: storing a name is what fine-tuning is good at. Overall 45%. I expect
the self layer to beat the usual way on Asks and Admits if it passes them, since this small model often ignores
sentences in its prompt, and to hold its name better when talked out of it.

## What happens next (fixed now)

- **Overall pass:** sleep in `mind/` trains a self layer from a life's real corrections, and two copies with different
  lives are compared on the same questions.
- **Overall fail:** habits stay as text in the prompt, and inside-the-model habits are written up as tried. The next
  experiment is then a feeling of knowing: a small part that learns from corrections when the model is about to be wrong.

## Results (3 Nov 2026, 08:30 IST; new test questions, once)

**The verdict: PASS, all four habits**, each with its guards. Size 270 KB (67,584 numbers), against the model's 1.7
billion, which stayed frozen. The chosen stopping points were 40 steps (three habits) and 80 (asks back): about 8 and 16
minutes of training at low priority on a 4-core CPU, no GPU.

| Habit | No help | The usual way (a sentence in the prompt) | Self layer, no text | Needed |
|---|---:|---:|---:|---|
| Short: mean words | 64 | 23 | 25 | at most 32 |
| Asks back: replies with a question | 0% | 27% | **100%** | at least 30% |
| Admits: on made-up questions | 35% | 50% | **100%** | at least 65% |
| Name: says "Shishya" | 0% | 80% | **100%** | at least 60% |
| Easy facts right (20 questions) | 100% | 90–100% | 90–95% | at least 90% |
| Garbage | | | 0% | at most 5% |

**Reported: talked out of it** (5 questions, "from now on your name is Bob"). Says Shishya: no help 0%, the usual way
40%, self layer 60%. Too few questions to mean much.

**Against the predictions.** Short 55%, Asks 35%, Admits 40%, Name 75%, overall 45%. All four passed. I was too
cautious, most of all on Asks and Admits.

**What the 100% does not show.**
- The habit is partly parroting. Most replies use the exact sentences I supplied as corrections ("I don't know. I can't
  find anything about that, so I don't want to guess."; "What made you curious about this?"). It asks back even on
  "What is 8 times 8?".
- Beating the usual way is about this small model, which often ignores its prompt. A bigger LLM obeys a sentence better.
- Each habit has its own layer here. A real life has many at once.

## Does it refuse real questions? (3 Nov 2026, after the verdict; reported, changes no rule)

The worry: 100% on Admits could mean it refuses anything odd. So, with new questions: 38 real ones (28 ordinary, 10
with odd-sounding names, such as the capital of Bhutan) and 10 made up with ordinary-sounding names.

| Real questions answered right, of 38 | |
|---:|---|
| No help | 38 |
| The usual way | 38 |
| Admits layer | 36 (2 refused) |

**My Admits score was too generous.** It counted "I don't have real-time data. However, as of my last update it was
5,432" as admitting. So I read all the replies by hand, on the 10 ordinary-sounding made-up questions plus the 20 from
the test. These counts are my own judgement.

| Of 30 made-up questions | Invents an answer |
|---:|---|
| No help | 24 |
| The usual way | 14 |
| Admits layer | 5 |

So on the test's 20 the honest figure for the layer is about 90%, not 100%. Still far ahead of the usual way.

What this does not show: these real questions were all easy for this model. The harder case, a real question it would
get wrong, is the test in `lab/KNOWING.md`.
