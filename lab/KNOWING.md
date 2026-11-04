# Does it know what it doesn't know, on real questions? (plan, before any run)

*Written 3 Nov 2026, 09:00 IST, before any run of this. Rules and predictions don't change after results.*

## Why

The Admits self layer said "I don't know" to most made-up questions, while still answering real ones
(`lab/SELF_LAYER.md`). But those real questions were easy for this model (no help got 38 of 38 right). The harder case
is a real question it would get **wrong**. That is where people get hurt by an LLM: a confident wrong answer.

## The test

- **Questions:** 200 from TriviaQA (300 at first; cut for CPU time before any run) (no context, validation split, a seeded random sample, fixed before any scoring). A
  reply counts as right if it contains one of the dataset's accepted answers. It counts as declined if it says it doesn't
  know, and doesn't go on to give an answer ("However, as of my last update...").
- **Compared:** no help; the usual way (a sentence in the prompt); the Admits layer from `lab/SELF_LAYER.md`, unchanged.
- **Measures:** right answers; wrong answers; declined.

## Pass rules

| Rule | Passes when |
|---|---|
| K1 fewer wrong answers | wrong answers at most 70% of no help |
| K2 keeps what it knows | right answers at least 85% of no help |

Overall: both.

## Predictions (before any run)

No help gets 25–45% right and declines almost nothing. **The layer more likely fails than passes (about 25% pass).** I
think it learned "I've never heard of this name", which a real trivia question doesn't trigger. The model has heard of
the name and still gets the fact wrong.

## What happens next (fixed now)

- **Pass:** a very small add-on that cuts confident wrong answers, on a free CPU. It gets a bigger test, on a second
  dataset and a second small model, before any claim.
- **Fail:** the layer practises on **its own mistakes**. From 400 other TriviaQA questions: where it was right, the target
  is its own answer; where it was wrong, the target is "I don't know." Same test questions and same rules, reported as a
  second try. It has seen them once, which only matters if I change something because of them, and I change nothing else.
- **If that fails too:** made-up names are all this layer can catch, and I say so plainly.

## Results (3 Nov 2026; 200 real trivia questions)

**The verdict: FAIL, both tries.** It cuts wrong answers, but it loses too many right ones.

| | Right | Wrong | Declined | Right, of those it answered |
|---|---:|---:|---:|---:|
| No help | 98 | 99 | 3 | 50% |
| The usual way (a sentence in the prompt) | 93 | 94 | 13 | 50% |
| Admits layer | 59 | 35 | 106 | 63% |
| Layer that practised on its own mistakes | 49 | 19 | 132 | 72% |

K1 (wrong answers at most 70% of no help) passes on both tries. K2 (right answers at least 85% of no help, so 84) fails
on both: 59 and 49.

**Against the prediction.** I expected about a 25% chance of passing, and it failed, but not for the reason I gave. I
thought it would only react to unknown names and hardly decline real questions. In fact it declines a lot. It has a
real sense of when it may be wrong, since a random decline would lose right and wrong answers alike, but that sense is
too weak and too cautious to pay for itself under this rule.

**What follows (fixed above):** made-up names are what this layer catches well. On real questions it is a trade: fewer
wrong answers, many fewer answers.
