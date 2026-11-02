# From v6 to a final Shishya: what the research says, and the plan (v7, v8, v9)

*Written 2 Nov 2026, 17:00 IST, while v6 runs. Each version still gets its own plan, pass rules and predictions,
committed before its runs. This file explains why these steps, in this order.*

## The goal (the owner's words)

"Not about beating RAG but creating something that is really very efficient, intelligent, useful, better
architecture to the humanity." And: "let's try to beat LLM like Pathway did."

So every version is judged on a scorecard, not on one rival:

| | What it means here |
|---|---|
| Learns after training | a fact read once is known, with no gradients and no retraining |
| Understands | asked in other words, by an outside examiner |
| Keeps up to date | a fact that changes is answered with its newest value |
| Doesn't forget | old facts and ordinary reading stay as good |
| Reasons | combines facts it learned at different times |
| Efficient | parameters, memory bytes per fact, ms per session, runs without a big GPU |
| Useful | works on real text and real questions, not only children's stories |

## Where Shishya stands (our own results)

- **"Beat an LLM like Pathway did" is already true at small scale.** Same size (12.8M) and the same 100M bytes of
  TinyStories: Shishya's Brain reads held-out text 7.5% better than GPT (0.544 against 0.588 bits per byte;
  `brain/RESULTS.md`). Pathway reports the same for BDH against GPT-2 from 10M to 1B parameters.
- **Learning after training: far ahead of an LLM.** After one reading, v5 on the Brain recalls 99% and answers in
  unseen wordings 96.5% (V5.md). The same-size GPT, fine-tuned on the same stream, recalls 21%.
- **Weak spots** (V6.md): changes said in other words don't replace the old value; five facts in one sentence blur;
  facts can't be chained without a plan; the memory is a fixed 268 MB whatever it holds; the Brain only knows
  children's stories.

## What the research says (2025–2026)

- **Pathway, BDH** ([paper](https://arxiv.org/pdf/2509.26507)): matches GPT-2-size transformers at 10M–1B
  parameters, with sparse, brain-like activity. In 2026 Pathway reports reasoning results (ARC-AGI-1, Sudoku) from
  larger BDH variants ([summary](https://www.codesota.com/baby-dragon-hatchling)). Vendor-reported, not replicated.
- **Google, Titans / ATLAS / Nested Learning (Hope)**
  ([blog](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning)):
  - memories updated at test time by surprise, which is what our delta rule does;
  - ATLAS fits the memory to a window of recent inputs, not one step at a time;
  - Hope uses a spectrum of memories, each updating at its own speed (fast → slow → weights).
- **Google, "Language Models Need Sleep"** ([arXiv 2606.03979](https://arxiv.org/pdf/2606.03979)): a wake–sleep
  cycle. Sleep moves short-term knowledge into the parameters; "dreaming" makes practice data.
- **Meta, sparse memory finetuning** ([arXiv 2510.15103](https://arxiv.org/abs/2510.15103)): learn new facts by
  updating only the few memory slots they use. Forgetting on an old task falls from 89% (full finetuning) and 71%
  (LoRA) to 11%.
- **Fast and slow learners** (SuRe, [arXiv 2511.22367](https://arxiv.org/html/2511.22367v1)): pick what to replay
  by surprise, and fold a fast learner into a slow one; the two parts help each other. This is the brain's
  "complementary learning systems": the hippocampus learns fast, the cortex slowly, and replay in sleep links them.
- **Chaining facts** ("two-hop curse", [arXiv 2411.16353](https://arxiv.org/abs/2411.16353)): LLMs too mostly fail
  to combine two separately learned facts without thinking out loud. Thinking out loud is the accepted route.
- **A bigger Brain exists for free:** [d3vmeh/bdh-100m-bytes](https://huggingface.co/d3vmeh/bdh-100m-bytes), MIT
  license, 100.9M parameters, Pathway's reference BDH, trained on about 5.4 GB of FineWeb-Edu (83%) and
  TinyStories (17%). Training that ourselves would take hundreds of hours of free GPU; this one only has to be
  checked.

## The plan, and why this order

The mechanisms are cheap to test on the small Brain; scale is expensive. So the mechanisms come first, and the
bigger Brain once, at the end.

### v7: sleep (keep up to date, don't forget, efficient memory)

Today all of life goes into one fixed 268 MB fast memory, one write at a time. Each write is a single
least-squares step (the delta rule), so later writes disturb earlier ones, and the memory never gets smaller.

- **Awake:** as v6. The fast memory learns each session by the local rule, and it also keeps the day's traces
  (who units, what vector, value) in a small log. These are numbers, not text.
- **Asleep, once a "day":**
  1. Replay the day's traces, together with the facts already in the slow memory.
  2. Where two traces have the same person and nearly the same meaning but different values, the later one wins:
     the old value is deliberately erased. This is v6's open problem, solved at sleep.
  3. Fit the slow memory to all of them at once, by several passes of the same local rule over shuffled traces
     (interleaved replay, as in the brain), instead of one pass in the order they came.
  4. Empty the fast memory.
- **The slow memory is sparse:** only units that hold something take space (Meta's lesson: touch only the slots a
  fact uses). Memory bytes grow with the facts known, not a fixed 268 MB.
- **Tests:** a 30-day life with thousands of facts and many changes. Each version runs with and without sleep:
  - current values;
  - old facts kept;
  - memory bytes per fact;
  - ordinary reading kept.
- **No gradients in life or in sleep.** Sleep is the local rule run offline, so the "learns without gradients"
  claim stays.

### v8: thinking (reason)

The Brain learns to think out loud on its own. A multi-step question makes it write its first step ("Zubak's best
friend is Kipos."), read that step back through the memory, and then answer.
- **Training:** a short new childhood for the Brain itself (gradients, done once, like any training), on the
  teacher's example thoughts. Train relations only, as before.
- **Test:** two- and three-step questions asked directly. Today these are at chance for every learner.

### v9: the bigger Brain, then the final Shishya (useful, and a fair fight with an LLM)

- **The reader:** the 100M BDH above, after checks:
  - it loads into Pathway's reference code;
  - its bits per byte on held-out FineWeb-Edu;
  - its license.
- **v5–v8's memory, sleep and thinking move onto it** (a new reader childhood, on the free GPU).
- **The fair fight** (the "beat an LLM" claim, with the rules fixed first):
  - Against GPT-2 small (124M) and SmolLM2-135M, on held-out reading and on learning after training, with each
    rival's usual way of learning (fine-tuning, RAG).
  - Against the same rivals on efficiency: memory per fact, ms per session, CPU only.
- **The final write-up:** what works, what doesn't, and what a person can use it for:
  - a personal memory that learns from a conversation once and stays up to date;
  - a small on-device model that learns without retraining.

### v9's main test: the student beats its teacher, in one area (added 2 Nov 2026, 21:00 IST, the owner's request)

The owner asked whether Shishya can take an LLM's knowledge quickly and then do better than that LLM. The honest
answer, which this test checks: not at everything (a 100M student can't hold what a billions-parameter teacher
holds), but in one area, after time has passed, yes, because Shishya keeps learning and the LLM is frozen.

**How the knowledge moves (fastest first).**
1. **Facts, through the memory:** the teacher LLM writes lessons; Shishya reads each once. No gradients; seconds.
2. **Skills, through the weights:** the teacher writes teaching text for the area; the 100M Brain gets a short
   training on it (distillation, as in TinyStories and Phi). Done once, on the free Kaggle GPU.

**The test (rules and numbers are fixed in V9.md before any run).**
- **The area:** real facts that changed after the teacher's training date, from public data (for example
  Wikidata: who leads a country, a company or a team; records; capitals and offices). Facts that didn't change are
  in it too.
- **Time 0:** the teacher (a free model) teaches Shishya the area as the teacher knows it.
- **Then life:** Shishya reads short news-style texts about the changes, once each, over many days. Corrections
  come from a quiz, as in the long life.
- **The exam (multiple choice, scored the same for all):**
  - the current value of facts that changed;
  - facts that never changed;
  - all of it in other wordings (an outside examiner).
- **Who takes it:**
  - the teacher as is (frozen);
  - the teacher with RAG over the same news;
  - a small LLM fine-tuned on the same news;
  - Shishya.
  Each also reports its efficiency: parameters, memory, ms per question, GPU or not.
- **"Beats its teacher" means** that Shishya is ahead of the frozen teacher on the full exam, by a 95% bootstrap
  lower bound above 0. It must also keep the teacher's knowledge: on never-changed facts it needs at least 90% of
  the teacher's score. RAG is reported next to it, as everywhere.
- **What it would show:** a small model on a CPU, taught once by a big LLM, then better than that LLM on what
  matters to its user, without retraining. It would not show that Shishya is a better chatbot.

## Risks I see now

- v6 may show the meaning part of the address can't be trained to put "moved to" where "lives in" is asked. v7's
  sleep rule (the later one wins) then does the work, but only if the two traces' meanings are close enough to be
  matched.
- The 100M BDH may not load cleanly, or may read worse than its card says. Then v9 trains a smaller general
  Brain (about 25M, FineWeb-Edu) over several free Kaggle sessions.
- Real names and news text are much harder than children's stories; the memory's address (built for short
  made-up names) may need work on them before v9's test is fair.
- A 100M Brain is still far from a useful assistant. The honest claim for v9 is a better *learning* architecture
  at small scale, not a better chatbot.
