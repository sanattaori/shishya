<div align="center">

# Shishya

### A neural network that keeps learning after training.

It reads a sentence once and knows it. No retraining, and no notes kept on the side.

[![CI](https://github.com/sanattaori/shishya/actions/workflows/ci.yml/badge.svg)](https://github.com/sanattaori/shishya/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanattaori/shishya/blob/main/brain/demo.ipynb)

**[Interactive guide: see how it works, with things you can click](https://sanattaori.github.io/shishya/)**

[How it works](#how-it-works) · [Is it an LLM?](#questions-people-ask) · [Results](#results) · [What we tried](#what-we-tried) · [Try it](#try-it) · [Full report](docs/RESEARCH.md)

</div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/architecture-dark.svg">
  <img alt="The life of the model. Birth: starts with random weights and learns language from scratch on 100M bytes of children's stories, with backprop. Childhood: language is frozen, and it learns how to use a memory, with backprop for the last time. Life: it reads and learns with no backprop. While it lives, it reads a sentence, neurons fire in a frozen network, the rare name and the meaning pick a place in the lasting memory, and later it can answer." src="docs/img/architecture-light.svg" width="100%">
</picture>

## The idea

Language models stop learning when training ends. There are two usual fixes:

- **Train again.** This is slow, and it damages what the model already knew.
- **Keep the text and paste it back in (RAG).** This works well, but the model itself has learned nothing.

Shishya tries a third way, closer to a brain: **reading a fact changes lasting connections inside the network.**

After one reading it scores 96%, where a same-size GPT, fine-tuned, scores 14%.

> **This isn't an LLM with something added.** It's a neural network we trained from random weights on children's stories,
> plus a lasting memory. There's no GPT-style LLM, stored text or search inside.

## How it works

### How it was trained

| Stage | What happens | Backprop? |
|---|---|---:|
| **Birth** | Starts with random weights. Learns language from scratch by reading 100M bytes of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) as raw bytes. 6 hours on a free GPU. | Yes |
| **Childhood** | Language is frozen. One small part learns how to use a memory, on made-up people and kinds of facts the tests never ask about. | Yes, for the last time |
| **Life** | It reads and learns. Only the memory changes. | **Never** |

### The parts

| Part | What it does | Changes in life? |
|---|---|---:|
| **Neural network** | Knows language. Reads one byte at a time | No |
| **Who** | The rare name picks 40 of the memory's 16,384 rows | No |
| **What** | The meaning picks a spot in those rows | No |
| **Lasting memory** | A fixed block of numbers that holds what it learned | **Yes. This is the learning.** |

### How it learns

Say it reads *"Priya lives in a town called Nashik."*

1. "Priya" is new to it, so the sentence is worth keeping, and the name picks the rows.
2. The network's sense of "a town comes next" picks the spot.
3. It writes the next letter there, harder when it is more surprised.

The whole learning rule fits on one line, with no gradients and no optimiser:

```text
M[row] += strength · who(row) · what ⊗ (next letter − what the memory says now)
```

This comes from **Hebbian learning**, the brain's "fire together, wire together": a connection changes using only what it
can see locally, with no error signal sent back through the whole network. Ours also corrects itself, writing only the
difference from what it already holds. This is called the **delta rule**.

### How it answers

Ask *"The town where Priya lives is called"*. The name opens the same rows. The words are different, but the meaning is
close, so it finds a nearby spot. What is stored there nudges the next letter towards **N**, then **a**, then **s**.

There's no database and no search. The answer is the network's normal guess, nudged by the memory.

### Inside the network: BDH

The network uses [BDH](https://arxiv.org/abs/2509.26507), Pathway's brain-like design. Where GPT passes dense lists of numbers
around, BDH passes neurons that fire or stay silent.

1. **Neurons fire.** Each byte switches on some neurons, and the rest stay silent.
2. **They link to the past.** Where the same neurons fired on an earlier byte, a temporary link forms.
3. **Neurons fire again,** this time from that context.
4. **Both must agree.** A neuron stays on only if steps 1 and 3 both fired it.
5. **Update and repeat.** The same weights run 6 rounds, like one brain thinking 6 times.

Those links vanish when the text ends, so BDH remembers nothing afterwards. **Shishya adds the part that lasts.**

Want more? The [interactive guide](https://sanattaori.github.io/shishya/) steps through this, and the
[full architecture](docs/ARCHITECTURE.md) has the shapes, equations and code.

## Questions people ask

**Is this an LLM?**
Not in the usual sense. No GPT-style LLM is inside, and it calls no LLM API. It's a small neural network that predicts the next
byte, plus a memory. Like any model that predicts text, it models language, and one of its steps is computed like attention.
But it's built from neurons that fire or stay silent, and it can't chat.

**Did you train it from scratch?**
Yes. Our 12.8M network started from random weights and read 100M bytes of TinyStories, in 6 hours on a free T4 GPU. It predicts
held-out text better than a same-size GPT on the same bytes (0.544 against 0.588 bits per byte; lower is better).

**Which network gives which result?**

| Network | Who trained it | Used for |
|---|---|---|
| **12.8M** | Us, from scratch | The 96% and 96.5% results, and the no-LLM runs up to v6 |
| **100.9M** | [Others](https://huggingface.co/d3vmeh/bdh-100m-bytes), free and MIT, on 5.4 GB of general English | v7's results and the demo. Ours only knows children's stories |

**Is there an LLM anywhere in this repo?**
Only outside the model, and always labelled: as baselines to compare against, in one test of the same memory on a small LLM, in
a side study, in v8's tests where a free LLM restates sentences first, and to write some test wordings.

**Does it keep the text?**
No. It keeps numbers in a fixed 268 MB block. That's much bigger than keeping the text, which is under 1 MB in our longest test.

**Is it smarter than ChatGPT?**
No. It can't hold a conversation. The research question is narrower: can a network keep learning after training?

## Results

The pass rules were fixed before each run. Each question is a choice among 5 names, so guessing scores 20%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/research-dark.svg">
  <img alt="Bar chart. One reading: Shishya 96%, fine-tuned GPT 14%. Unseen words: Shishya 96.5%, RAG 96%. Changed facts after five days: Shishya 74.2%, RAG-3 33.3%. Change said in other words: Shishya 45.6%, RAG-3 47.5%. Five facts in one sentence: Shishya 39%, RAG-3 68.5%. Two facts combined, no thinking out loud: Shishya 27.5%, RAG 31%." src="docs/img/research-light.svg" width="100%">
</picture>

**What works**
- After one reading it scores 96%. A same-size GPT, fine-tuned, scores 14%.
- It scores 96.5% on wordings it never saw, about kinds of fact it never trained on.
- Over five days, with its memory saved and reloaded each day, it kept 88% of its unchanged facts.

**What doesn't work yet**
- A change said in new words often doesn't replace the old fact.
- Several facts in one sentence blur together.
- It can't combine two facts unless it thinks out loud.

**Should you use this instead of RAG? Not today.** RAG ties or beats it on most tests. RAG's text is also far smaller:
under 1 MB, against a fixed 268 MB. And Shishya has only been tested on made-up names in simple sentences. This is
research, not a product. [The report](docs/RESEARCH.md#limits-of-this-work) lists every limit we know of.

## What surprised us

1. **Where you write matters more than the network.** The same network, with only a new way to choose where facts are
   stored, went from 67% to 99%.
2. **One idea fixed other wordings.** Splitting the memory's address into *who* and *what* took unseen wordings from 35% to
   96.5%.
3. **A network eight times bigger didn't fix the hard problem.** It understood kinds of fact much better, yet updating facts
   didn't improve. The limit is in the memory.
4. **Sleep made it worse.** Replaying memories, as a brain does in sleep, protected old facts but brought old values back.
5. **Forgetting isn't the same as updating.** Protecting stored facts stopped all forgetting, and then new values won only
   3% of the time.
6. **We often guessed wrong.** We predicted 45–65% for one test, and it scored 98%. Writing guesses down first is how we
   know.

## What we tried

Eight versions, on free hardware. Each plan and pass rule was written down before its test ran.

| Version | What we tried | Result |
|---|---|---|
| **v1, v2** | Train the network from scratch, and add lasting connections | Fail. Recall 35%, then 67%, and it lost old facts |
| **v3** | Choose where to write by the sentence's rare words | 99%, and 96% after one reading. Failed only on speed |
| **v4** | Add meaning and protection. Includes two ideas tried from IBM's Larimar, which the final model doesn't use | Fail. Other wordings stuck at 35% |
| **Hybrid** | Test the same memory on a small LLM, against RAG | Fail. Other wordings 27% |
| **v5** | Separate *who* from *what* | **All rules pass.** Other wordings 96.5% with no LLM |
| **v6** | A harder exam: changing facts, five facts in one sentence | Fail, at 34–65%. But a five-day life passes |
| **v7** | A bigger network, plus sleep | Fail. Changes 46%. The five-day life passes again |
| **v8 tests** | A free LLM restates sentences first | Fail by one question. Changes 69%, needing 70% |
| **v8, latest** | The same, and hide replaced values | Fail by two questions. Changes 85%, the best yet, but unchanged facts 81% |
| **Side study** | A 270 KB layer that gives a frozen LLM habits | Mixed. Each habit passes alone; real trivia fails |

The project began as a small [rules library](docs/DECISIONS.md), which is still here.

## How we kept it honest

- **Rules first.** Each plan, pass rule and prediction was committed before its run. A version one question short still
  fails.
- **Failures stay.** Most versions failed, and all of them are [on the record](docs/RESEARCH.md#every-version).
- **Numbers you can check.** The raw outputs are in [`brain/results/`](brain/results/).

## Try it

**In your browser:** the [interactive guide](https://sanattaori.github.io/shishya/#demo) ends with a small toy you can teach.
It's a tiny version to show the idea, and it isn't the real model.

**The real model** runs on a normal CPU, with no GPU or API key. The first run downloads the free 100.9M network (400 MB).

```bash
git clone https://github.com/sanattaori/shishya && cd shishya
pip install torch numpy huggingface_hub

python -m brain.teach --tour      # what it can do, and what it can't
python -m brain.teach             # teach it yourself
```

```text
> Zubak has a dog named Rex.      a line ending in "." is taught
> The name of Zubak's dog is      anything else is a sentence for it to finish
```

Use rare names, and expect a few seconds per letter. Only the unfinished sentence goes in, so the answer can only come
from its memory. The red lines are real failures, and we left them in.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/teach-dark.svg">
  <img alt="Terminal replay. Shishya reads three sentences once, then finishes them from memory, including in words it has not read. Two failures are shown as well." src="docs/img/teach-light.svg" width="100%">
</picture>

## Find your way around

| Where | What |
|---|---|
| [Interactive guide](https://sanattaori.github.io/shishya/) | The easiest place to start |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How it works, in plain words and in full detail |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | The report: results, lessons, limits and open problems |
| [`docs/README.md`](docs/README.md) | A guide to every document |
| [`brain/v5.py`](brain/v5.py) | The memory: address, read, write and training |
| [`lab/`](lab/) | A side study: a 270 KB layer that gives a frozen LLM habits |
| [`shishya/`](shishya/) | A separate library: the most likely answer that breaks no rule. [Docs](docs/DECISIONS.md) |

**Want to help?** The [open problems](docs/RESEARCH.md#open-problems) are small, clear experiments. See
[CONTRIBUTING](CONTRIBUTING.md).

## Credits

Built on [BDH](https://arxiv.org/abs/2509.26507) by Pathway ([code](https://github.com/pathwaycom/bdh), MIT) and the free
[100M BDH](https://huggingface.co/d3vmeh/bdh-100m-bytes). Our network is trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Baselines use
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

*Shishya* (शिष्य) is Sanskrit for "student": one who learns from a *guru*.

## Citation

```bibtex
@misc{taori2026shishya,
  author = {Taori, Sanat},
  title  = {Shishya: a neural network that keeps learning after training},
  year   = {2026},
  url    = {https://github.com/sanattaori/shishya}
}
```

Licensed under [Apache-2.0](LICENSE).
