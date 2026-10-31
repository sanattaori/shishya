# Where the work is (hand-off for a new session)

*Last updated 31 Oct 2026, 17:15 UTC. Branch `claude/bold-ride-lsozd7`. Read this, then
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

## Where it stands (31 Oct 2026, 17:15 UTC): both verdicts are in, nothing is running

| Run | Kaggle kernel | Verdict |
|---|---|---|
| v1: GPT, BDH, Brain v1 (100M bytes each) + life test | `sanattaori/shishya-brain-life` | FAIL (L1, L2, L4); L3 passes |
| v2 Brain (100M bytes) | `sanattaori/shishya-brain-v2` | childhood 0.544 bits/byte |
| v2 life test against v1's GPT and BDH | `sanattaori/shishya-brain-life2` | FAIL (L1, L2, L4); **L3 passes by +45 points** |

**Headline (brain/RESULTS.md section 3):**
- v2 learns facts from reading alone: 67% after 3 exposures, 66% after one, with no gradients and no damage to its
  reading. GPT and BDH fine-tuned on every session get 21–22%.
- It misses the 80% bar.
- It keeps only half of the early facts over 636 sessions. The likely cause is interference from writing every step.
- It takes about 0.5 s per session on a CPU.

The small outputs (`life.json`, `diagnose.json`, metrics, logs) are committed in `brain/results/`. The trained
models are not in git. Re-download them with `kaggle kernels output sanattaori/<kernel> -p brain/out/kaggle-<kernel>` and a
token from the owner. The Kaggle code dataset is `sanattaori/shishya-brain-code` (private). The GitHub repo is
private, which is why Kaggle gets the code that way.

**Credentials:** a session needs `KAGGLE_API_TOKEN` from the owner. Never write it into the repo or a log. The
owner deletes temporary tokens after use.

## Next steps (decide with the owner first)

1. **Choose what v3 changes:** see the ideas below, plus "write only surprising steps", a lower write rate, and more
   units.
2. **Write v3 into docs/PLAN.md as a dated amendment before any v3 run.**
   - Same rules unless the owner agrees to change L4's speed target.
   - Same test stream, so v1, v2 and v3 stay comparable.
   - Reuse v1's GPT and BDH, so only the Brain is retrained (`run.sh … --brain2` style).
3. **Check it first on the CPU** (tiny models, dev names only): `brain/diagnose.py`, `brain/probes.py`, the short
   life test.
4. **Then the full run.** On free T4s that is about 6 h for the Brain plus 1 h of life test. A rented A100 does it in
   about 1 h.

Environment for a fresh container:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" numpy kaggle
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt     # only in Claude Code cloud containers
```

## Ideas for a v3 (with the owner)

0. **Write less, so the memory keeps more.**
   - The fix: write only the most surprising steps (a threshold), or a lower base rate.
   - The evidence: v2 writes every step of every text, about 160,000 writes over a life. Its dev recall rose as
     the write rate fell (72% → 80%), and 0.5, the lowest rate tried, was the best.
   - Directly aimed at L1 and L2.

1. **Less forgetting over long lives.**
   - What: each synapse gets harder to change the more it has stored (metaplasticity). It's a local, diagonal form
     of Larimar's recursive least-squares write, which protects memory directions already in use.
   - Why: aims at rule L2.
2. **Forgetting on request:** a `forget` call (Larimar writes with α = −1) for wrong or private facts.
3. **Questions in other words.**
   - What: a second, meaning-based address beside v2's wording-based one.
   - Why: v2 is at chance on paraphrases (17%); Larimar gets 83–88% with BERT addresses.

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
