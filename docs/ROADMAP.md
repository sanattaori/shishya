# After v3: Shishya goes to school (a design, for the owner to decide)

*Proposal, 1 Nov 2026. Nothing here is a pass rule yet. Each step becomes a dated amendment to
[PLAN.md](PLAN.md) before its run, with its rules and predictions fixed first, as before.*

## Where v3 leaves us

| | v3 (test stream) |
|---|---:|
| Facts recalled after 3 readings / after 1 | 99% / 96% |
| Earliest facts kept over 636 sessions | 95% |
| Corrections / changed facts | 90% / 77% |
| Questions in other words (chance 20%) | 34% |
| Fine-tuned GPT and BDH, same size | 21–22% |
| CPU time per 256-byte session | 665 ms (rule L4: 50 ms) |

v3 passes every learning rule. By the rules fixed in advance, the overall verdict is still FAIL, on L4 (speed).
That stays in the record. L4's 50 ms was too fast for BDH at this size from the start. A realistic speed target is
set before the next run. **Proposal:** ≤ 1 s per 256-byte session on a 4-core CPU, with GPU time reported too.

## Three things to be clear about first

**1. Brains forget too, and that's useful.**
- People lose most new details within days unless they are repeated, rewarded or slept on (Ebbinghaus, 1885).
  Nobody remembers lunch on a Tuesday three weeks ago.
- What lasts is what matters: repeated things, rewarded things, surprising things, things replayed in sleep.
- Forgetting frees room and stops old details from getting in the way.
- **So the goal is not "never forget". It is "keep what matters, forget on purpose".**
- v3 already keeps 95% of its earliest facts. It forgets for one reason: its memory is a fixed size
  (16,384 units), so every new write disturbs old ones a little. Sleep and growth (below) are the brain's
  answers to that.

**2. Where intelligence comes from.**
- **Childhood (size and data):** language, reasoning skill, general knowledge. Ours is 12.8M parameters trained
  on children's stories, so it is a child. A smarter Shishya needs a bigger childhood.
- **The lasting memory:** learning after training, which LLMs can't do. v3 has this.
- **A teacher and sleep:** turn experiences into understanding.

**3. A teacher LLM is allowed. An LLM inside the student is not.**
- The owner's rule stays: the student is Shishya Brain, never an LLM.
- But a student can have a teacher. The Guru is an LLM that *talks to* the student: it gives lessons,
  explains the same thing in other words, asks quizzes and corrects mistakes.
- The student learns only through its own local rule, as from any other text. Its synapses change; the Guru's
  never enter it.
- That is exactly the name: guru and shishya.
- (TinyStories, its childhood text, was itself written by GPT-4. So Shishya already learned from a teacher once.)

## The flagship test: "a year of school in one night"

**The setup.**
- A **made-up world** is written as a hidden fact graph: people, places, pets, jobs, events, with made-up
  names. No model can know it in advance, and every answer can be checked exactly.
- The **Guru** teaches it in lessons: short texts in varied wording, then quizzes. The student answers, the Guru
  checks against the graph, and wrong answers get a correction (reward 3).
- The **student** lives through the school with its local rule only, plus sleep (below).
- A **closed-book final exam**, with no Guru and nothing in context:

| Exam part | Example |
|---|---|
| Facts taught once | "Zubak's cat is called" → Fugez |
| In other words | "What is the name of the cat that belongs to Zubak?" |
| Two steps | "Where does Zubak's best friend live?" (best friend → Kipos; Kipos → Zifig) |
| Changed facts | Fugez ran away; Zubak's new cat is Pelin |
| Forgotten on request | `forget` a fact: it's gone, and its neighbours are not |
| Still reads well | held-out bits per byte |

**Compared with, all at the same size and the same memory budget in bytes:**
1. GPT fine-tuned on the lessons.
2. GPT + retrieval (keep the lesson texts, look them up). This is the honest baseline: it's what people use today.
3. The Guru itself (the upper bound).

**The claim to aim for, written so it can fail:**
> A student under 100M parameters, running on a laptop CPU, learns a whole made-up world from a teacher, with no
> gradients after childhood and a memory of fixed size. Closed-book, it answers ≥ 90% of facts taught once,
> ≥ 80% asked in other words, and ≥ 60% of two-step questions. It updates changed facts and forgets on request.
> It beats fine-tuning by a wide margin and matches retrieval at the same memory budget.

**Is that new?**
- Parts of it exist:
  - Larimar: one-shot edits and forgetting, through an encoder.
  - MemoryLLM and M+: a self-updating memory inside an LLM.
  - Titans: memorising at test time with gradients.
  - Memorizing Transformers and kNN-LM: retrieval inside the model.
  - Sparse memory finetuning; "Language Models Need Sleep".
- I don't know of one system that does all of it: local rules only, learning from a teacher, two-step use of
  what it learned, sleep, forgetting, and fixed memory, on a laptop.
- **A careful literature check comes before we say "first".**

## The brain, part by part (what to build)

1. **Fast memory (the hippocampus): v3's, plus reconsolidation.**
   - Metaplasticity: used synapses become stable.
   - Reconsolidation: a correction reopens them.
   - On dev, together with v3's address: changed facts 93% (v3 83%), one reading 100%.
2. **Meaning address (semantic memory), for questions in other words.**
   - Each word's code drifts slowly, by a local rule, toward the codes of words seen in the same places
     (random indexing, Kanerva 2000). "Belongs to" and "'s" end up close.
   - The address also gets an order-free part (which words, not where), so "the cat of Zubak" and "Zubak's cat"
     share it.
   - No BERT, nothing trained by backprop after childhood. v3 is at 34% here; this is the main weak spot.
3. **The Guru (teacher), outside the brain.**
   - It writes lessons from the hidden graph, rephrases them, quizzes, and corrects.
   - It can be any chat model. The `shishya` package already talks to OpenRouter and Claude Code
     (`shishya/guru.py`).
   - **Rough cost:** 10,000 facts × 3 lessons × 200 tokens is about 6M tokens. That's a few dollars on a small
     model.
4. **Thinking out loud, for two-step questions.** The student writes the first step ("Zubak's best friend is
   Kipos."). That text cues its own memory for the second step ("Kipos lives in a town called … Zifig"). It
   needs a childhood with chain drills, so a new childhood.
5. **Sleep, so memory doesn't fill up.**
   - At night it replays what it stored (cued from its own memory), moves it into the slow weights, then
     clears room in the fast memory. That is what the brain's hippocampus does with the cortex.
   - Two variants, for the owner to choose:
     - **(a) Local rules only.** New, and hard. Nobody has a good recipe yet.
     - **(b) Gradient replay offline,** like "Language Models Need Sleep". It works today, but it uses
       gradients at night.
6. **Growth (new neurons).** Add memory units when the old ones are heavily used, so capacity grows with life.
7. **A smarter core.**
   - A bigger childhood: 50–100M parameters, and better data (teacher-written textbooks, as TinyStories was).
   - Note: v3's memory address doesn't depend on BDH's neurons. The same memory can sit on any network,
     including a faster one. BDH costs about 7× GPT's compute per byte (v1 canary). The owner chooses.

## The steps (each one pre-registered)

| Step | What it adds | Test | Rough cost |
|---|---|---|---|
| **v4** | Meaning address; reconsolidation; a retrieval baseline; L4 target reset | Today's life test + questions in other words as a rule; 3 seeds | This CPU + free Kaggle, 2–3 days |
| **v5** | The Guru school, first version: a made-up world of 1,000 facts, varied wording, quizzes | A closed-book exam (facts, other words, changed, forget) | A few $ of API; free Kaggle |
| **v6** | Thinking out loud; two-step questions (new childhood with chain drills) | Two-step exam | A rented GPU, about $20–60 |
| **v7** | Sleep and growth; a long life (10,000+ facts) | Long exam: does it keep the first lessons? | Free Kaggle or about $20 |
| **v8** | A bigger core and the flagship test above | The full school, against fine-tuning, retrieval and the Guru | A rented GPU, about $100–300 |

The costs are rough guesses from the speeds measured so far (T4: 4,511 bytes per second for the 12.8M Brain).
They get measured in each step's canary.

## Decisions for the owner

1. **The new L4 target** (proposal: ≤ 1 s per session on a 4-core CPU).
2. **Sleep:** local rules only, gradient replay at night, or both compared?
3. **The Guru:** which model, and what monthly budget?
4. **The core:** keep BDH, or allow a faster network under the same memory?
5. **A rented GPU** for v6 and v8 (about $1–3 an hour).

## What doesn't change

- The student never contains an LLM. After childhood it learns only by local rules (and sleep, if chosen).
- Rules and predictions are fixed before every run, and the test stream is used once.
- Failures are reported as they come.
