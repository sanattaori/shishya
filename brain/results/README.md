# Raw results

The small output files of every run, so the numbers in [../RESULTS.md](../RESULTS.md) can be checked and
re-analysed without downloading anything. The trained models (≈160 MB per run) are not here. They stay in the Kaggle
kernels' outputs: `kaggle kernels output sanattaori/<kernel> -p brain/out/kaggle-<kernel>`, with a token from the
owner.

| Folder | Run | Kaggle kernel | Code |
|---|---|---|---|
| `v1/` | GPT, BDH, Brain v1 (100M bytes each), then the life test | `shishya-brain-life` | `commit.txt` (ba36374) |
| `v2/` | Brain v2 training, then its life test against v1's GPT and BDH | `shishya-brain-v2`, `shishya-brain-life2` | `train-commit.txt` (0b0964d), `life-commit.txt` (09f8841) |
| `canary/` | 20 steps per model and a short life (a GPU check, not a result) | `shishya-brain-canary` | ba36374 |
| `cpu/` | the tiny CPU checks (not results): short runs, the long tiny v1 and v2, v2's short life and diagnostic | this machine | as in RESULTS.md |

- `life.json`:
  - the dev runs (every setting tried)
  - the chosen settings and the test-stream results
  - `main_correct`: one true/false per main fact, for the paired bootstrap
  - the verdict per rule and the CPU check
- `life.log`: the life test's printout.
- `diagnose.json`: the memory diagnostic (a made-up name two texts later; dev names).
- `*-metrics.json`: one childhood run: parameters, bytes, held-out bits/byte, speed, the full training history.
- `*-train.log`: the training printout.

**Known problems in these files:**
- **v1 and the canary, `resources.peak_rss_mb`:** wrong, inflated by the parent process. See RESULTS.md: v1's real
  peak is 996 MB.
