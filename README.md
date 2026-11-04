<div align="center">

# Shishya

### A neural network that keeps learning after training.

It reads a sentence once and knows it. No retraining, and no notes kept on the side.

[![CI](https://github.com/sanattaori/shishya/actions/workflows/ci.yml/badge.svg)](https://github.com/sanattaori/shishya/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanattaori/shishya/blob/main/brain/demo.ipynb)

**[Architecture](#the-architecture)** · **[Results](#results)** · **[Try it](#try-it)** · **[Full report](docs/RESEARCH.md)**

</div>

## The idea

Language models stop learning when training ends. There are two usual fixes:

- **Train again.** This is slow, and it damages what the model already knew.
- **Keep the text and paste it back in (RAG).** This works well, but the model itself has learned nothing.

Shishya tries a third way, closer to a brain: **reading a fact changes lasting connections inside the model**, using a
simple local rule. After one reading it scores 96%, where a same-size GPT, fine-tuned, scores 14%.

## The architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/architecture-dark.svg">
  <img alt="Architecture. The text goes to a frozen BDH reader and to the who part of the address. The reader gives the what part. Who and what address a lasting memory, the only part that changes. The memory nudges the next letter." src="docs/img/architecture-light.svg" width="100%">
</picture>

| Part | What it does | Changes after training? |
|---|---|---:|
| **Reader** | A language model that already knows English | No |
| **Who** | The rare name picks 40 of the memory's 16,384 rows | No |
| **What** | The meaning picks a spot in those rows | No |
| **Lasting memory** | A fixed block of numbers that holds what it learned | **Yes. This is the learning.** |

### How it learns

Say it reads *"Priya lives in a town called Nashik."*

1. "Priya" is new to it, so the sentence is worth keeping, and the name picks the rows.
2. The reader's sense of "a town comes next" picks the spot.
3. It writes the next letter there, harder when it is more surprised.

The whole learning rule fits on one line, with no gradients and no optimiser:

```text
M[row] += strength · who(row) · what ⊗ (next letter − what the memory says now)
```

### How it answers

Ask *"The town where Priya lives is called"*. The name opens the same rows. The words are different, but the meaning is
close, so it finds a nearby spot. What is stored there nudges the next letter towards **N**, then **a**, then **s**.

There's no database and no search. The answer is the reader's normal guess, nudged by the memory.

### Inside the reader: BDH

The reader is [BDH](https://arxiv.org/abs/2509.26507), Pathway's brain-like language model. Where GPT passes dense lists
of numbers around, BDH passes **65,536 neurons that fire or stay silent**.

1. **Neurons fire.** Each byte switches on some neurons, and the rest stay silent.
2. **They link to the past.** Where the same neurons fired on an earlier byte, a temporary link forms and brings that
   information forward. It's "fire together, wire together", computed as attention.
3. **Neurons fire again,** this time from that context.
4. **Both must agree.** A neuron stays on only if steps 1 and 3 both fired it.
5. **Update and repeat.** The same weights run 6 rounds, like one brain thinking 6 times.

Those temporary links vanish when the text ends, so BDH remembers nothing afterwards. **Shishya adds the part that lasts.**

**[Read the full architecture](docs/ARCHITECTURE.md)**: plain words first, then the shapes, equations and code.

## Results

The pass rules were fixed before each run. Each question is a choice among 5 names, so guessing scores 20%.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/research-dark.svg">
  <img alt="Bar chart. One reading: Shishya 96%, fine-tuned GPT 14%. Unseen words: Shishya 96.5%, RAG 96%. Changed facts after five days: Shishya 74.2%, RAG-3 33.3%. Change said in other words: Shishya 45.6%, RAG-3 47.5%. Five facts in one sentence: Shishya 39%, RAG-3 68.5%. Two facts combined, no thinking out loud: Shishya 29%, RAG 31%." src="docs/img/research-light.svg" width="100%">
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

## Try it

It runs on a normal CPU. You don't need a GPU or an API key. The first run downloads a free 400 MB model.

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

## How we kept it honest

- **Rules first.** Each plan, pass rule and prediction was committed before its run. A version one question short still
  fails.
- **Failures stay.** Eight versions, and most of them failed. All of them are [on the record](docs/RESEARCH.md#every-version).
- **Numbers you can check.** The raw outputs are in [`brain/results/`](brain/results/).

Want to help? The [open problems](docs/RESEARCH.md#open-problems) are small, clear experiments. Issues and pull requests
are welcome.

## Find your way around

| Where | What |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How it works, in plain words and in full detail |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | The report: results, lessons, limits and open problems |
| [`brain/v5.py`](brain/v5.py) | The memory: address, read, write and training |
| [`brain/teach.py`](brain/teach.py) | The demo |
| [`lab/`](lab/) | A side study: a 270 KB layer that gives a frozen LLM habits |
| [`shishya/`](shishya/) | A separate library: the most likely answer that breaks no rule. [Docs](docs/DECISIONS.md) |

## Credits

Built on [BDH](https://arxiv.org/abs/2509.26507) by Pathway ([code](https://github.com/pathwaycom/bdh), MIT) and the free
[100M BDH](https://huggingface.co/d3vmeh/bdh-100m-bytes) model. Baselines use
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). Training uses
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

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
