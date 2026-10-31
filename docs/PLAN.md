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
- TinyStories: Eldan and Li (2023).
