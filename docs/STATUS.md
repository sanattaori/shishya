# Where the work is (hand-off for a new session)

*Last updated 2 Nov 2026, 04:00 IST. Branch `brain-v5` (the `ccr-ebda0c39-9rnnt2` work + `brain-hybrid` + v5). For v5
read [V5.md](V5.md) first. Before v5: read this, then [PLAN.md](PLAN.md) (the design, the pass
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
| v4 life test (v2's network, v4's memory) | `sanattaori/shishya-brain-v4` | L1, L2, L3 pass; **FAIL on L5** (held-out wordings 35% < 50%); L4 reported only |
| Hybrid stage 1 (SmolLM2-135M + v3's memory; [HYBRID.md](HYBRID.md)) | CPU | H1–H3 pass; FAIL on H4 (vs RAG) and H5 (held-out wordings 27%) |
| **v5, 135M (main; [V5.md](V5.md))** | CPU, `brain/out/v5/` | **all five pass**: other words 89.5%, held-out wordings 93%; RAG ties or wins most rows |
| v5, SmolLM2-1.7B (scale) | `sanattaori/shishya-brain-v5` | all five pass: other words 93%, held-out wordings 98% |
| **v5 with no LLM (the BDH Brain as reader)** | `sanattaori/shishya-brain-v5brain` | **all five pass**: other words 96.5%, held-out wordings 99% |
| The hard school ([V6.md](V6.md): changing facts, five facts in one sentence) | CPU, `brain/out/v6/` | **FAIL all four** (135M): changes 19% vs RAG-3 47.5%; one sentence 35.5% |

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

**The owner's decisions, 1 Nov 2026** (on ROADMAP.md's open questions):
- "Plan and start." Do what is right; commit often (a new session continues from [RESUME.md](RESUME.md)).
- **Speed is set aside for now.** L4 is reported, not a pass rule, from v4 on.
- **Add intelligence and test it.**
- The owner wants to understand sleep; it is explained in ROADMAP.md ("How sleep works").
- The Guru's model and budget, the core network, and a rented GPU are left to the work. There is no LLM API key
  in the cloud container, so the Guru (v5) needs a key from the owner.

**v4, in order (tick as done):**
1. [x] Held-out wordings (`data.HELDOUT_PARAPHRASE`, committed before step 2): a second way to ask each kind of question, used only on the test stream, so v4's
   "questions in other words" can't be fitted to the wordings seen on dev.
2. [x] The meaning part of the address (`key_bag`; probes in brain/RESULTS.md section 7): the sentence's words without their order, each counting by how new it is,
   bound to the word being read. Probe it on dev facts (`python -m brain.probes`).
3. [x] Dev lives (`python -m brain.study`; brain/RESULTS.md section 7; v4 = meaning part 4 + metaplasticity + reconsolidation): v3 + meaning part, with and without metaplasticity +
   reconsolidation. Choose by a rule written before the last result, as for v3.
4. [x] A retrieval baseline in the life test, plus an exact-lookup one (dev: 26% and 100%): GPT reads the best-matching stored sentence before each
   question. Reported, not a rule: it shows where Shishya stands against "keep every text and look it up".
5. [x] The v4 amendment (docs/PLAN.md, 1 Nov), canary4, and the v4 verdict (brain/RESULTS.md section 8):
   L1–L3 pass; L5 fails (35% in held-out wordings, as predicted); changed facts 90%; one reading 76%.
6. [x] **v5 (2 Nov): done, see [V5.md](V5.md).** A frozen LLM plus a who ⊗ what memory, written by the
   delta rule with no gradients. Passes all five rules at 135M and 1.7B. Credentials are in the repo's gitignored
   `.env` (`KAGGLE_API_TOKEN`, `OPENROUTER_API_KEY`); load with `set -a; . ./.env; set +a`. Free OpenRouter
   models (`:free`) only. **Next (V5.md's rule, all pass):** tests where keeping text and looking it up
   struggles (many facts in one sentence, facts that change many times, combining facts without a plan), then
   sleep. **Done for the first two: [V6.md](V6.md), v5 fails them.** Next is v6: a childhood with changes and
   many-fact sentences in it, and an address that keeps a person's facts apart. The history below is kept for reference.

   *(Before v5, 1 Nov:)* The school is started: `brain/school.py` (a made-up world, a template Guru, quizzes with
   corrections, a closed-book exam with plain, reworded and two-step questions; dev names only). First runs:
   `brain/out/school/v3.json`, `v4.json` (brain/RESULTS.md section 9): after one lesson per fact, plain questions
   100%, but unseen wordings and two-step questions at chance (each person has 5 facts, and the name-led meaning
   part can't tell which one is asked). With 3 lessons per fact in 3 wordings: still at chance (24%), but v4's
   protection keeps plain recall at 100% where v3 falls to 80%. An address probe shows v4 *finds* the right fact
   for 76% of unseen wordings but shares only ~30% of its address, so the read is too faint (RESULTS.md section 9).
   "Who × what" and a linear fingerprint read-back did not help.

   **The plan for v5 (my recommendation, 1 Nov):** a new childhood. In childhood the read path only ever saw a
   fact asked again in the same words, so it never learned to use a partial match. v5's childhood drills restate
   facts **in other words** (and, for two steps, chain two facts: "Zubak's friend is Kipos. Kipos lives in …").
   - The rephrasings come from the Guru. An open LLM (Qwen) can run on Kaggle's GPU, so no API key is needed;
     templates work as a start.
   - The memory is v4's. Childhood is about 6–7 h on Kaggle's free T4 (as v2's was), or about 1 h on a rented A100.
   - The test is the school's exam plus the life test, with rules fixed first.

   To do, in order:
   - **v5's childhood drills:** a pool of rephrasings per relation, split train / dev / test like the names, so
     the test wordings are never seen in childhood.
   - **The meaning part must carry what is asked, not only about whom** (the school's main finding).
   - A word code that keeps letter order (ROADMAP.md "Learned on the way"): fewer traces after `forget`, less
     overlap.
   - A dev rule that also looks at one-reading recall (v4's rule picked rate 4.0, and one reading fell to 76%).
   - Questions in other words, the main gap (35% held-out). Ideas: word codes that drift toward words seen in the
     same places (random indexing, a local rule); lessons from the Guru in many wordings.
   - Two-step questions need "thinking out loud" and a new childhood (a rented GPU).
   - An LLM Guru needs an API key from the owner (OpenRouter or Anthropic).

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
- **Scoring 256 questions at once takes about 8 GB on a CPU** (the BDH activations). Two such runs at once on
  this 15 GB machine get killed for memory. `study.py` and `school.py` score 64 at a time.
- **Several study lives at once share 4 cores.** Their "ms per session" are then 2–3× too high. Only a run
  alone (or Kaggle's CPU check) measures speed.
- **Kaggle shows no logs while a kernel runs.** Use `run.sh … --canary3` first for GPU-only bugs.
- **Don't edit `run.sh` while a copy of it is running:** bash reads the script as it goes.
- **Mounted Kaggle runs carry an old copy of the code.** `kernel.py` now takes code only from the
  `shishya-brain-code` dataset.
- **BDH is slow per byte** (about 7× GPT's compute at the same size).
