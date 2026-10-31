# Shishya Brain (research)

A small network that **keeps learning after training**, the way a brain does: born with backprop, it lives with
a local Hebbian rule. The plan, and the pass rules fixed before any run, are in [docs/PLAN.md](../docs/PLAN.md).

- `models.py`: GPT (baseline), plain BDH (Pathway's maths, checked against `third_party/bdh.py`), and the Brain
  (BDH + lasting synapses written by a surprise- and reward-weighted delta rule).
- `data.py`: TinyStories as bytes, made-up facts with names split into train / dev / test, childhood episodes,
  the life stream.
- `childhood.py`: trains one model. Every model sees the same bytes.
- `lifetest.py`: the life test and the verdict.
- `diagnose.py`: does the Brain use its memory at all (short gaps, dev names)? A diagnostic, not a rule.
- `probes.py`: can a question find its own fact's address? The probes behind the v2 amendment.
- `report.py`: the tables for `RESULTS.md`.
- `kaggle/`: the full run on Kaggle's free GPU (`--canary` first: a 15-minute GPU check).

Results so far: [RESULTS.md](RESULTS.md). Where the work is and how to resume: [docs/STATUS.md](../docs/STATUS.md).

This is research code outside the `shishya` package. It needs PyTorch: `pip install torch numpy`.

## Run

```bash
# the CPU check: tiny models, a short life (numbers here are not the verdict)
for m in gpt bdh brain; do
  python -m brain.childhood --model $m --preset tiny --smoke --bytes 2.5e6 --lr 2e-3 --warmup 30 --out brain/out/tiny/$m
done
python -m brain.lifetest --runs brain/out/tiny --smoke --small --out brain/out/tiny/life.json

# the real test, on Kaggle (push the branch first)
brain/kaggle/run.sh YOUR_KAGGLE_USERNAME

# tests
python -m pytest brain/tests -q
```

## Credits

BDH: A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz, *The Dragon Hatchling: The Missing
Link between the Transformer and Models of the Brain* (arXiv 2509.26507). Code © 2025 Pathway Technology, MIT
licence, vendored unchanged in `third_party/` (commit 2b0d7a4). TinyStories: R. Eldan and Y. Li (CDLA-Sharing-1.0),
downloaded at run time, not redistributed.
