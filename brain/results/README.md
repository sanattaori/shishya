# Raw results

The small output files of every run, so the numbers in [../RESULTS.md](../RESULTS.md) can be checked and
re-analysed without downloading anything. The trained models (≈160 MB per run) are not here. `brain/kaggle/run.sh`
rebuilds them. v7's reader is in `brain/weights/`.

| Folder | Run | Kaggle kernel | Code |
|---|---|---|---|
| `v1/` | GPT, BDH, Brain v1 (100M bytes each), then the life test | `shishya-brain-life` | `commit.txt` (8c85a34) |
| `v2/` | Brain v2 training, then its life test against v1's GPT and BDH | `shishya-brain-v2`, `shishya-brain-life2` | `train-commit.txt` (56ae8d1), `life-commit.txt` (d34e446) |
| `canary/` | 20 steps per model and a short life (a GPU check, not a result) | `shishya-brain-canary` | 8c85a34 |
| `v5-135M/`, `v5-1.7B/`, `v5-brain/` | v5 on each reader: school, life and baselines ([V5](../../docs/V5.md)) | `shishya-brain-v5`, `shishya-brain-v5brain` | |
| `v6/`, `v7/` | The hard school and the long life ([V6](../../docs/V6.md), [V7](../../docs/V7.md)) | `shishya-brain-v6brain`, `shishya-brain-v7` | |
| `v8-poc/` | v8's one-hour tests, in progress ([V8_POC](../../docs/V8_POC.md)) | this machine | |
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

**Commit ids in the raw files.** `v1/commit.txt`, `v2/train-commit.txt` and `v2/life-commit.txt` are kept as the runs wrote them, so they use the ids from before author names were made uniform. Today's ids are: ba36374 = 8c85a34, 0b0964d = 56ae8d1, 09f8841 = d34e446.
