# Where the work is (hand-off for a new session)

*Last updated 31 Oct 2026, 14:20 UTC. Branch `claude/bold-ride-lsozd7`. Read this, then
[PLAN.md](PLAN.md) (the design and the pass rules), then [brain/RESULTS.md](../brain/RESULTS.md).*

## How we got here (the owner's decisions, in order)

1. **The owner wanted a new direction for Shishya:** something new, needed, and close to the name.
2. **"Compile repeated LLM tasks into tested programs" was planned, then set aside.** The owner: anyone can ask an
   LLM for a program; they wanted an intelligent model, not hard-coded code. The plan is in
   [parked/compile-to-program.md](parked/compile-to-program.md).
3. **A small LLM that learns from corrections was rejected too.** The owner: no next-token LLMs, because they don't
   keep learning.
4. **Chosen: a brain-like network that keeps learning after training.** Born with backprop, it lives with a local
   Hebbian rule.
   - It builds on Pathway's BDH (MIT code, vendored in `brain/third_party/`).
   - It adds lasting synapses written by a surprise- and reward-weighted delta rule.
   - The owner knows the idea isn't unique (Pathway, Google's HOPE, "Language Models Need Sleep", Meta's sparse
     memory finetuning). They chose it as **an open, independent test** of "a brain-like model that keeps
     learning", plus learning from corrections inside the synapses.
5. **The owner asked for plain English and honest results,** with pass rules fixed before runs. That matches how
   they run their trading bot.

## What is running

| What | Where | Expected |
|---|---|---|
| v1 verdict run: GPT, BDH, Brain v1, then the life test | Kaggle kernel `sanattaori/shishya-brain-life` | ~17:00 UTC 31 Oct |
| v2 Brain training | Kaggle kernel `sanattaori/shishya-brain-v2` | before or around then |
| tiny v2 Brain: done (memory saves 3.41 bits/byte on names, v1 0.02; see brain/RESULTS.md); its short CPU life test running | this session's CPU (`brain/out/tiny-long/`, not in git) | ~15:00 UTC; only a check |

Kaggle runs on the owner's account (`sanattaori`). The code reaches Kaggle as the private dataset
`sanattaori/shishya-brain-code`, because the GitHub repo is private. **Credentials:** a session needs
`KAGGLE_API_TOKEN` from the owner. Never write it into the repo or a log. The owner deletes temporary tokens after use.

## Next steps

```bash
# 0. a fresh container: environment
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" numpy kaggle
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export KAGGLE_API_TOKEN=...                       # ask the owner
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt     # only in Claude Code cloud containers

# 1. are the runs done?
.venv/bin/kaggle kernels status sanattaori/shishya-brain-life
.venv/bin/kaggle kernels status sanattaori/shishya-brain-v2

# 2. when both are COMPLETE: the v2 life test (mounts both outputs; about an hour)
KAGGLE=.venv/bin/kaggle brain/kaggle/run.sh sanattaori --life2

# 3. download and report
.venv/bin/kaggle kernels output sanattaori/shishya-brain-life -p brain/out/kaggle-shishya-brain-life
.venv/bin/python -m brain.report brain/out/kaggle-shishya-brain-life/out      # v1 verdict
.venv/bin/python -m brain.report brain/out/kaggle-shishya-brain-life2/out     # v2 verdict

# 4. v1's CPU memory with the fixed (chunked) scoring: the life rate is life.json["chosen"]["brain"]
.venv/bin/python -m brain.lifetest --resources --runs brain/out/kaggle-shishya-brain-life/out --life-rate <rate>
```

Then:
- Put the tables and verdicts in `brain/RESULTS.md`. Say plainly which rules failed and why: the diagnostics in
  `life.json` and `diagnose.json`.
- Tell the owner in plain words. If a rule failed, decide the next step *with* them; don't tune on the test stream.

## Open questions for the owner

- **Rule L4 says ≤ 50 ms per session on a CPU. The `bee` Brain takes about 625 ms**, which is still faster than a
  person reads. It stays as written for these runs. A different target can be set only before a future run.
- **Rent a fast GPU (A100/H100, $1–3 an hour) for the next round?** These runs take 6–10 h on free T4s. The owner
  asked about multiple Kaggle accounts: no, against Kaggle's terms.

## Things that bit us (so they don't again)

- **Background commands in these sessions stop after 2 h.** Start long CPU runs detached (`setsid nohup … &`),
  then poll. `childhood.py` now saves the model at every evaluation.
- **Kaggle shows no logs while a kernel runs.** Use `run.sh … --canary` first (15 min) for GPU-only bugs.
- **Don't edit `run.sh` while a copy of it is running:** bash reads the script as it goes. Pollers are inline loops.
- **BDH is slow per byte** (about 7× GPT's compute at the same size). The v1 Brain also learns through its writes.
- **The v2 Kaggle run trained with the older, slower window-key code** (32 shifted sums). The life test uses the
  convolution. Both give the same keys; a test checks it.
