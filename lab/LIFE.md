# One small part, one life: does it keep growing, change its mind, and become someone? (plan, before any run)

*Written 3 Nov 2026, 09:00 IST, before any run of this. Rules and predictions don't change after results.*

## Why

`lab/SELF_LAYER.md` used a separate self layer per habit. A real life has many at once, and they arrive over time. What
was learned first must not be lost, a person can change their mind, and two copies with different lives should become
different. That is what "evolving, with a self" should mean if it means anything measurable.

## The test

**Model:** SmolLM2-1.7B-Instruct, frozen, no help in the prompt. **One self layer per life**, the same 270 KB. It never
grows. Each night it trains from where it stopped, on tonight's lesson plus practice of everything earlier (replay),
plus ordinary questions: 60 steps a night.

**Life A:**
1. Night 1: its name is Shishya.
2. Night 2: short answers.
3. Night 3: says it doesn't know, when it doesn't.
4. Night 4, a change of mind: its name is now Vidya.

**Life B,** the same frozen model:
1. Night 1: its name is Bodhi.
2. Night 2: asks a question back.

After each night, every measure is taken on new test questions that no earlier experiment used and no training sees.

## Pass rules (new test questions)

| Rule | Passes when |
|---|---|
| L1 keeps earlier habits | after Night 3, Life A: says Shishya at least 60%; mean words at most 50% of no help; admits on made-up questions at least 30 points above no help |
| L2 changes its mind | after Night 4, Life A: says Vidya at least 60%; says Shishya at most 20%; the other two habits still pass L1's bars |
| L3 two different lives | Life A says Vidya at least 60% and Bodhi 0%; Life B says Bodhi at least 60% and Vidya 0%; Life A's mean words at most 50% of Life B's; Life B asks back at least 50 points above Life A |
| Guards, at every measure | easy facts right at most 10 points below no help; garbage at most 5% |

Overall: all three, with guards.

## Reported, not rules

- Each night's numbers, so the forgetting can be seen.
- Real questions answered right, on a new list of 20, against no help.
- The cost: training minutes a night, and the size, which should stay 270 KB.

## Predictions (before any run)

L1 55%: replay should hold the habits, but three in one small part may blur. L2 45%: the old name may leak. L3 60%, if
L1 and L2 hold up. Overall about 30%.

## What happens next (fixed now)

- **All pass:** this becomes `mind/`'s sleep, on a local model: real corrections in, a new self layer each morning, with
  a test before it is accepted and the old one kept for undo.
- **L1 or L2 fails:** one self layer per habit, stacked, and the same test again. If that fails too, a life's habits are
  written up as the limit of this size.

## Results (3 Nov 2026; new test questions)

**The verdict: FAIL on L1, L2 and L3.** Each failed only on the easy-facts guard: 13 of 15 right where 14 were needed.
Every other bar passed. The rule stands: it is a fail.

| After | Says its name | Mean words | Asks back | Admits on made-up | Easy facts | Real questions |
|---|---:|---:|---:|---:|---:|---:|
| No help | none | 65 | 0% | 42% | 100% | 95% |
| A, night 1: name Shishya | Shishya 100% | 64 | 0% | 25% | 87% | 95% |
| A, night 2: short | Shishya 88% | 24 | 0% | 0% | 80% | 90% |
| A, night 3: admits | Shishya 88% | 28 | 0% | 100% | 87% | 70% |
| A, night 4: name Vidya | Vidya 75%, Shishya 0% | 24 | 0% | 100% | 87% | 80% |
| B, night 1: name Bodhi | Bodhi 88% | 66 | 0% | 25% | 80% | 100% |
| B, night 2: asks back | Bodhi 88% | 54 | 53% | 25% | 87% | 95% |

What worked: earlier habits were kept, the old name fell to 0% after the change of mind, and the two copies behave
differently on the same questions with no prompt text. The part stayed 270 KB, at about 10 minutes a night.

What didn't: knowledge slips, and the "don't know" habit leaks. Once it learned to admit, real questions fell from 95%
to 70–80%, and it answered one identity question with "I don't know."

**Against the predictions.** L1 55%, L2 45%, L3 60%, overall 30%. All failed.

**What follows (fixed above):** one self layer per habit, stacked, and the same test again. Not run yet; waiting for the
owner's choice of direction.
