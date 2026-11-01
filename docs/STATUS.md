# Where the work is (hand-off for a new session)

*Last updated 31 Oct 2026, 23:35 UTC. Branch `ccr-ebda0c39-9rnnt2`. Read this, then [PLAN.md](PLAN.md) (the design, the pass
rules and the dated amendments), then [brain/RESULTS.md](../brain/RESULTS.md). How v3 works: [brain/ARCHITECTURE.md](../brain/ARCHITECTURE.md).*

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
6. **31 Oct, evening: v3.** The owner:
   - Does not want Shishya turned into an LLM or bolted onto one. The goal is a model that **beats LLMs at
     learning after training**, through our own ideas.
   - Asked for v3 with new ideas for the unsolved problems, "like the IBM (Larimar) ones".
   - Agreed a decision point: if v3 passes L1 and L2, go on to v4; if not, write up v1–v3 and stop this line.
   - Did not change L4 (50 ms per session), so it stays as written.
   - Asked why BDH isn't popular (new, unproven at scale, slower on GPUs: our T4 runs give 8,240 bytes/s against
     GPT's 62,058). The answer they got: the reason to continue is learning after training, not RAM or speed.

## Where it stands

| Run | Kaggle kernel | Verdict |
|---|---|---|
| v1: GPT, BDH, Brain v1 (100M bytes each) + life test | `sanattaori/shishya-brain-life` | FAIL (L1, L2, L4); L3 passes |
| v2 Brain (100M bytes) | `sanattaori/shishya-brain-v2` | childhood 0.544 bits/byte |
| v2 life test against v1's GPT and BDH | `sanattaori/shishya-brain-life2` | FAIL (L1, L2, L4); L3 passes by +45 points |
| v3 life test (v2's network, v3's memory) | `sanattaori/shishya-brain-v3` | **L1, L2, L3 pass**; FAIL on L4 (CPU speed) only |

**What v3 is** ([PLAN.md](PLAN.md), amendment of 31 Oct): v2's trained network, unchanged, with a new way to address
and write its lasting memory. The settings are in `brain/models.py` (`V3`); `brain/make_v3.py` builds it from the
v2 run.

**Why** (brain/RESULTS.md sections 4–5, dev stream only):
- v2 learns every fact and loses them later, because facts of the same kind share most of their address.
- v3's address is the words of the current sentence, each counting by how new it is. Same-kind facts then share
  13% of their address instead of 61%.
- On the dev life this takes recall from 80% to 99%, and the earliest facts kept from 0.50 to 0.95 of their
  mid-life level.

**Outputs** are not in git. They are in `brain/out/` in the session that ran them. Re-download with
`kaggle kernels output sanattaori/<kernel> -p brain/out/kaggle-<kernel>`. The study lives are in
`brain/out/study/*.json`; compare them with `python -m brain.study --table brain/out/study/*.json`.

**Credentials:** a session needs `KAGGLE_API_TOKEN` from the owner. Never write it into the repo or a log. The
owner deletes temporary tokens after use.

## Next steps

**v3's verdict (brain/RESULTS.md section 6):**
- 99% recall after 3 readings, 96% after one. GPT and BDH fine-tuned: 21–22%.
- Early facts kept 0.95 of 1.00; held-out bits/byte −2.7%.
- Fails only L4: 665 ms per session on Kaggle's CPU against 50 ms (1.2 GB memory: OK).
- By the decision point agreed with the owner (pass L1 and L2), the work goes on to v4.

1. **Read [ROADMAP.md](ROADMAP.md) (1 Nov): "Shishya goes to school".** It is the proposed path from v4 to a flagship test, with the owner's five open decisions at the end. Choose v4 with the owner, in their direction (better learning than an LLM, no LLM inside; a teacher LLM outside is fine):
   - **Questions in other words** (v3: 34% on test, chance 20%). A meaning-based address without BERT: words
     that occur in the same places get similar codes, learned slowly in life. This is v3's main weak spot.
   - **Changed facts** (v3: 77%). Metaplasticity + reconsolidation got 93% on dev, against v3's 83%
     (brain/RESULTS.md section 5).
   - **Sleep:** move the lasting memory into the slow weights, so the memory can be emptied and the model keeps
     what it learned.
   - **Speed for L4:** the forward pass is about 50 billion operations per session, estimated. BDH's ReLU
     neurons are mostly zero, and sparse kernels could skip them. Or a smaller core under the same memory.
   - **Harder tests:** longer lives, many more facts, facts that must be combined, a bigger model against a GPT
     of the same size. Also a retrieval baseline ("keep every text and look it up"), so the comparison is honest.
2. **Write v4 into docs/PLAN.md as a dated amendment before any v4 run,** with the same discipline: dev stream
   only, rules and predictions fixed first.

Environment for a fresh container:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" numpy kaggle
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt     # only in Claude Code cloud containers
# the models and data: download the life2 kernel's output, then
mkdir -p brain/out/runs && for k in gpt bdh brain2; do ln -sfn ../kaggle-shishya-brain-life2/out/$k brain/out/runs/$k; done
mkdir -p brain/out/data && mv brain/out/kaggle-shishya-brain-life2/data/* brain/out/data/
```

## Open questions for the owner

- **L4 (≤ 50 ms per session on a CPU)** stays as written. The `bee` Brain's forward pass alone takes about 0.4 s
  on a 4-core CPU. A different target can be set only before a future run.
- **Rent a fast GPU (A100/H100, $1–3 an hour)** if a v4 needs new childhoods? The free T4s take 6–10 h each.

## Things that bit us (so they don't again)

- **Background commands in these sessions stop after 2 h.** Start long CPU runs detached
  (`setsid nohup … &`), then poll.
- **`pgrep -f` / `pkill -f` match the shell running them** when the pattern is in its own command line. A
  `pkill -f name` once killed its own command. Use the PID from `ps`.
- **Several study lives at once share 4 cores.** Their "ms per session" are then 2–3× too high. Only a run
  alone (or Kaggle's CPU check) measures speed.
- **Kaggle shows no logs while a kernel runs.** Use `run.sh … --canary3` first for GPU-only bugs.
- **Don't edit `run.sh` while a copy of it is running:** bash reads the script as it goes.
- **Mounted Kaggle runs carry an old copy of the code.** `kernel.py` now takes code only from the
  `shishya-brain-code` dataset.
- **BDH is slow per byte** (about 7× GPT's compute at the same size).
