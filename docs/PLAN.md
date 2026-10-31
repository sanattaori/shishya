# Shishya 1.0: practice makes programs

*A plan, written 31 Oct 2026. Nothing here is built yet. Every number marked "target" is a goal, not a result.*

## The decision

Shishya stops being a filter on top of a decision model and becomes a **practice engine**.

It watches an LLM do one of your tasks many times. Then it writes a program that does the same task: readable
Python plus a few tiny learned parts. It tests that program on your labels and runs it on a CPU in under a
millisecond. Your rules always hold. The program answers on its own only where it has proven itself, with a stated
error bound. Everything else still goes to the LLM, and each of those answers becomes more practice. After one
correction, it never makes that exact mistake again.

> **LLMs think from scratch every time. Shishya practises until it doesn't have to.**

Why this direction, in five lines:

1. **New as a product.** The closest work is two 2026 papers and two tiny repos. None proves itself on your labels,
   states an error bound, keeps your rules or learns from corrections ([what exists](#what-exists-today)).
2. **Big, plain need.** Most LLM calls in production repeat the same prompt with a new input. People pay for them in
   money (inference is now a top AI budget line), in seconds (1–10 s a call) and in trust (a different answer each run).
3. **Doable with what we have:** a CPU, Claude Code as the programmer, free open models as labellers, a free Kaggle
   GPU for the rare heavy part. No pretraining, no GPU cluster.
4. **It is the name.** A *shishya* learns from a *guru* by practice, and at the one thing practised, becomes better
   than the guru: faster, steadier, never forgetting a correction.
5. **It reuses all of v0.3:** fail-closed rules, the A\* rule search, the learner with held-out promotion,
   calibration, receipts, the Guru and the Gate. They become the trust layer of the new engine.

## Why not the other ideas

Each idea we discussed, judged honestly:

| Idea | Verdict | Why |
|---|---|---|
| Compress a flagship LLM so it beats everyone on a CPU | No | Flagship weights (Claude, GPT, Gemini) are closed. Open ones are already squeezed by large projects (llama.cpp, Unsloth, 1.58-bit BitNet-style models). A compressed model is still a model: smaller, but slower and worse than the original, never better. |
| A new way to predict the next token (diffusion, multi-token, non-transformer) | No, not for us | Proving a new architecture means pretraining on billions of tokens: $100k+ of GPUs. Big labs already ship diffusion and multi-token models. We can't win a compute race. |
| Fine-tune a small model on the LLM's answers | A part, not the product | Several companies sell this already. It is still a neural net on a GPU: it can't be read, it can't be bounded, and it can't follow rules. |
| A small model that learns live from every correction | Inside the plan | In general this is an open research problem. For one narrow task it works today: memory, repair and held-out promotion. v0.3 already has the first two. |
| Text that is guaranteed to follow rules (constrained decoding) | No | Outlines, XGrammar and llguidance already do it well. |
| Replaying an agent's tool calls (a behaviour cache) | Not now | muscle-mem / Butter do it. A replay can't handle a request it hasn't seen. |
| AGI, or better than LLMs at everything | No | No small team can reach it. "Better than the LLM *at your task*" is reachable and measurable, and it's what people pay for. |
| **Compile the LLM's task into a tested program** | **Yes** | The rest of this document. |

The "extraordinary result on a CPU" comes from **not calling the model at all**, not from shrinking it. A 50-line
program does in about 20 µs what an LLM call does in about 2 s. That is 100,000× faster (a target, to be measured).

## The idea, and why it is not just a cache

People learn a skill in three stages (Fitts and Posner, 1967): first slow and spelled out in words, then practised,
then automatic. John Anderson called the last step **knowledge compilation** (1982): what you first worked out step
by step becomes a fast procedure. LLMs never reach stage three. They work out every answer from scratch, even on
the millionth request of the same kind.

- **System 2** (chat and reasoning LLMs) thinks, slowly.
- **System 1** (decision models such as Jev and Kev) guesses fast, but you can't read the guess and it has no bound.
- **Shishya adds the missing stage: the habit.** It is fast like System 1, and you can read it and test it like
  ordinary code.

How a Shishya skill compares with the usual alternatives:

| | Repeats seen answers | Handles new inputs | Can be read | Error bound | Rules hold | Learns from 1 correction |
|---|---|---|---|---|---|---|
| Cache / semantic cache | yes | no / guesses | yes | no | no | yes (exact) |
| Fine-tuned small model | yes | yes | no | no | no | no (retrain) |
| The LLM itself | yes | yes | no | no | no | no |
| **Shishya skill** | yes | yes (it's a program) | yes, each branch names the examples behind it | yes, measured on held-out data | yes, fail closed | yes |

### Where the student can beat the guru

These are the ways a compiled skill can be **more accurate** than the LLM it learned from, not only faster:

1. **Votes, not one sample.** The skill learns from the majority of several answers, from more than one model
   family. One LLM call is one noisy sample. A student of the majority beats a single sample. Google's
   "distilling step-by-step" (2023) got a 770M model to beat 540B PaLM on ANLI. ReaComp (2026) got programs to beat
   best-of-K sampling of gpt-oss-120b by 16 points.
2. **Corrections stick.** A corrected case becomes a test, and the program must pass every test. An LLM repeats the
   same mistake forever.
3. **Rules remove a whole class of errors.** In v0.3, Kev-0.8B broke a rule 38% of the time; with Shishya, 0%.
4. **Same input, same answer.** No random errors from sampling.
5. **It knows what it doesn't know.** Below its proven confidence it hands the input to the guru. So it is never
   worse than the guru on the hard cases, and on the cases it answers it has a measured edge.
6. **Your conventions.** Company-specific labels and edge cases the guru doesn't know become explicit code.

Where it can't beat the guru: open-ended reasoning, world knowledge that never appeared in the traffic, and brand-new
events. That is what handing hard cases to the guru is for.

## What exists today

What we found in October 2026 ([sources](#sources)):

- **Compiled AI** (arXiv 2604.05150, Apr 2026). An LLM writes code once for enterprise workflows, and the code then
  runs with zero model calls. It reached 96% task completion on BFCL, used 57× fewer tokens over 1,000 runs, and
  scored 80% on invoice key fields (DocILE). Its public pages describe no held-out test, no confidence, no fallback
  to an LLM and no learning. The repo has 4 stars.
- **ReaComp** (arXiv 2605.05485, May 2026). Coding agents turn about 100 reasoning traces into symbolic solvers.
  With no LLM at test time, the solvers beat best-of-K gpt-oss-120b by 16 points on PBEBench-Hard. Running the
  solver first with an LLM fallback cut tokens by 78%. This is the strongest evidence that a student program can beat
  the model it learned from. It is research code for puzzle-like domains.
- **rote** (7 stars). Compiles a Claude skill into a pipeline, but keeps the LLM for exactly the steps we want to
  compile: parsing, classifying, drafting.
- **Butter / muscle-mem** (YC W25). Cache and replay agent behaviour.
- **Labelling functions written by LLMs** (ScriptoriumWS 2025, DataSculpt 2023, Alfred 2023). LLMs write keyword
  and pattern rules to label training data. They are used to build training sets, not to serve answers, and come with
  no bound.
- **Conformal risk control** (Angelopoulos et al. 2023, with 2026 follow-ups). Pick a confidence threshold on
  held-out data so that, with high probability, the error on the answered cases stays below a target. This is the
  basis of our error bound.
- **Evolutionary code search** (FunSearch, AlphaEvolve, the open-source OpenEvolve and CodeEvolve). An LLM proposes
  program changes and a test scores them. We use the same loop, with "agrees with the labels on held-out data" as the
  score.

**The gap:** no open tool compiles the *judgment* part of an LLM call (classify, extract, route) into a program,
proves it on your labels, bounds its error, keeps your rules and keeps learning. Those five things are what Shishya
already does for decisions.

## Who needs it

- **Anyone who sends the same prompt with new inputs many times:** support ticket routing, intent detection, parsing
  documents and bank SMS, KYC fields, log triage, content tags, moderation, news triage, agent tool permissions.
- **Cost.** A 2026 survey of 240 enterprises found that median inference spend grew 4.1× in 2025. It is a vendor
  survey, so it is weak evidence, but it agrees with everything else we found.
- **Speed.** A voice reply or a trading decision can't wait 2 s.
- **Trust and audit.** The EU AI Act's high-risk duties (documentation, human oversight) now start on 2 Dec 2027.
  A program with tests and a receipt is far easier to document than a prompt.
- **Us.** kite-signal-bot's watchman sends news headlines to free OpenRouter models. On 7 Oct, Nemotron took
  80–160 s per batch. The second model family answered 4 of 160 calls in a week. A compiled triage skill would answer
  in about a millisecond and never hit a rate limit.

## Architecture

```text
             ┌──────────────── practice (offline, minutes) ────────────────┐
  traffic →  │ Recorder → Labellers (open models, 3 votes) → Splits        │
             │   → Programmer (Claude Code) writes solve.py                 │
             │   → Sandbox runs it → counterexamples → Programmer repairs   │
             │   → fit tiny parts → calibrate → certify → pack              │
             └──────────────────────────────────────────────────────────────┘
                                         │  skill pack: code + tiny weights + tests + receipt
                                         ▼
  request → Skill runtime → candidate answers + confidence → Rules (v0.3 search, fail closed)
                  │
                  ├─ confident and familiar → answer + receipt          (µs–ms, one CPU core)
                  └─ otherwise → Guru LLM (told the rules) → answer → recorded for the next practice

  audit:        2% of confident answers are re-asked in the background → live error bound → demote if over
  corrections:  memory at once → a test added → repair → re-certify → new version (rollback kept)
```

### 1. The task contract

One file says what the task is, who teaches it and how much error you accept:

```yaml
name: headline_impact
input:  {headline: string, company: string}
output:
  impact:   {enum: [bad, good, neutral]}
  material: {type: boolean}
guru:
  prompt: prompts/headline.md          # the prompt you use today
  labellers: [openrouter:qwen…:free, openrouter:nemotron…:free]   # whose answers it learns from
  votes: 3
  programmer: claude                   # who writes the code (Claude Code, your login)
rules: rules.json                      # v0.3 JSON / Python / Rego rules, unchanged
trust:
  max_error: 0.02                      # on the answers the skill gives on its own
  confidence: 0.95                     # how sure we must be of that bound
  audit_rate: 0.02                     # share of live answers re-checked by the guru
```

**Output types in 1.0:**
- enums and booleans (as in v0.3)
- numbers (exact or within a tolerance)
- strings with a format (date, money, ID)
- **strings copied from the input:** checked by code to appear in the input, so the skill can't make up text
- lists of objects (receipt line items, scored by set F1)

Free text (summaries, emails) is not compiled in 1.0. The guru keeps writing it.

### 2. Recorder

The Recorder logs every call: input, answer, who answered (skill or guru), model, cost, time, branch and version. It
is SQLite, like v0.3's `learn.db`. Splits are made by input text with a leak check, as in `train/news_specialist`.

### 3. Labellers: whose answers the skill learns from

- **Default:** open-weight models whose licence allows training on their outputs. They give 3 votes, from at least
  2 model families, batched 20 items per request (v0.2 measured that batching didn't hurt: 0.783 vs 0.736 macro-F1).
  When they agree, that is the label. When they disagree, the item is marked hard and goes first to a person.
- **People:** corrections, plus a short labelling session on the hard items. **Only human labels can promote a
  skill.**
- **Claude or GPT as labellers** only when you switch it on and your terms allow it. Anthropic's and OpenAI's terms
  limit using their outputs to build competing models. That is why v0.3 marks Guru answers `trainable: false`, and
  1.0 keeps that flag. Writing code with Claude is ordinary use, and that's its role here.

### 4. The programmer

Claude, through Claude Code (`claude -p`, your own login), writes and repairs `solve.py`. Each round it gets:
- the task contract and 30 examples
- its own notes
- the current program
- the program's worst counterexamples, each with the input, the expected answer, what the program said and from
  which branch

It answers with a small diff. A program looks like this:

```python
def solve(x, h):
    t = x["headline"].lower()
    if re.search(r"\bauditor\b.*\b(resign|quit)", t):
        return out(impact="bad", material=True, why="auditor resigns")   # branch b1: 23/23 on train
    if m := MONEY.search(t):
        ...
    label, p = h.classify("impact", t)      # a tiny learned part decides what the code can't
    return out(impact=label, material=None, p=p, why="classifier")
```

**What a program may use:** a subset of builtins; `re`, `math`, `datetime`, `unicodedata`, `string`,
`collections`, `itertools`; and the helpers `h.classify`, `h.score`, `h.similar`, `h.table`.

**What it may not use:** `import`, `open`, `exec`, `eval`, `getattr`, names starting with `_`, the network or files.

### 5. The search: counterexample-guided evolution

1. **Seeds:** the programmer writes notes on the task, then 4 programs with different strategies (rules first,
   parser first, mostly learned parts, a decision list).
2. **Score** each program in the sandbox on the training split: agreement with the labels, how much it can answer,
   size (code + table entries) and speed.
3. **Keep** a small population on the best trade-off between agreement, coverage and size, so a short program isn't
   thrown away for a slightly better long one.
4. **Mutate:** a parent goes back to the programmer with its 10 worst counterexamples and 3 cases it gets right (so
   it doesn't break them). This is counterexample-guided synthesis.
5. **Refit** the tiny parts after each change (well under a second, as in v0.3's learner).
6. **Against memorising:** a size penalty, and every table entry must be backed by at least 2 training examples or a
   human correction. Choices are made on the calibration split, never on the training split.
7. **Stop** when the budget is spent or the score stops improving. Target: ≤ 300 programmer calls per task.

### 6. Tiny learned parts

The tiny parts handle what code can't spell out. They are trained only on labels we may train on (people, and
labellers whose licence allows it):

- hashed n-gram logistic regression (already in `learn.py`; trains in under a second in pure Python)
- static word embeddings (model2vec-style: about 30 MB, numpy only, about 0.1 ms) for meaning
- nearest-neighbour memory over those embeddings
- optional: the v0.2 ModernBERT news specialist (17 ms) when a field needs it

### 7. Confidence, handing over to the guru, and the error bound

- **Branch scores.** Every `return` is a numbered branch, and its accuracy on calibration data is known ("412 of
  415").
- **Confidence** is a calibrated mix of:
  - the branch's accuracy
  - agreement among the 3 best programs
  - the tiny part's probability
  - how unlike the training inputs this input is
- **The threshold** is chosen on the calibration split with a Clopper–Pearson upper bound. With 95% probability, the
  error on the answers the skill gives stays ≤ `max_error`. That holds only for traffic like the calibration traffic,
  which is why the next point exists.
- **Live audit.** In the background, `audit_rate` of the confident answers are also sent to the guru. A running
  bound is kept. If it goes over `max_error`, the skill is demoted (everything goes to the guru), you are told, and it
  practises again.
- Below the threshold, or on an input unlike anything it practised on, the input goes to the guru.

### 8. Rules: unchanged from v0.3

The program and its tiny parts give the top few candidates per field. The v0.3 A\* search picks the most likely
combination that breaks no rule. Rules fail closed, and the same rules check the guru's answers. The only extension:
the search also runs over candidate lists for numbers and strings.

### 9. Learning from corrections

1. A person corrects an answer. **Memory** gives that exact input the right answer at once.
2. The case becomes a **test**. Tests are never deleted.
3. A **repair** job gives the programmer the failing case. Its diff must pass every test and must not lower the
   calibration score.
4. The skill is **re-certified** and becomes a new version (the last 10 are kept for rollback, as in v0.3).
5. **Promotion:** a version is used only when the skill plus the guru fallback beats or ties the guru alone on the
   frozen human-labelled set (paired bootstrap, the v0.3 code).

### 10. The skill pack

A skill is a folder you can diff in git and review like any code:

```text
headline_impact/
  solve.py        the program, each branch commented with the examples behind it
  parts.json      tiny learned parts
  memory.jsonl    exact corrections
  tests.jsonl     every correction + a sample of agreed cases (the regression suite)
  receipt.json    labellers, programmer, data hashes, scores with intervals, threshold, max_error, version
```

### 11. Ways to use it

- **Python** (1.0):
  ```python
  @shishya.skill("skills/headline_impact")
  def impact(headline: str, company: str) -> Impact: ...
  ```
  Before the skill is certified, it calls the guru, like a normal LLM function, and records the traffic.
- **CLI:** `shishya practice | certify | status | correct | serve`.
- **Later:** an OpenAI-compatible proxy that groups calls by prompt template and practises each group, so an app
  gets it with no code change.
- **The Gate stays.** Its approvals and rejections can become a skill later.

### 12. Security

- **While practising:** programmer-written code runs only in a sandbox: a separate `python -I` process with CPU and
  memory limits, after an AST allowlist check, and with no network where the OS allows it (Linux `unshare -n`).
- **When loading a pack:** the same AST check runs, plus a hash check against the receipt.
- **Over HTTP:** no user code, as in v0.3.

### What changes in the repo

- **Keep:** `rules.py`, `when.py`, `search.py`, `engine.py`, `guru.py`, `learn.py`, `calibrate.py`, `gate/`
- **Extend:** `schema.py` (numbers, formatted strings, copied strings, lists)
- **New:**
  - `shishya/practice/` (`record.py`, `label.py`, `sandbox.py`, `program.py`, `evolve.py`, `certify.py`)
  - `shishya/skill.py` and `shishya/pack.py`
  - `bench/practice/`

## The realistic goal

**North star:** for any repeated LLM task, Shishya answers most calls itself, at least as well as the LLM, about
1,000× faster, with a stated error bound, and never breaks a rule.

### Proof of concept: three public tasks with human labels, plus one unseen check

| Task | Data | What it tests |
|---|---|---|
| A. Headline impact | SEntFiN Indian headlines (human labels, MIT; already prepared in `train/`) | Meaning-heavy judgment, the hardest case for code |
| B. Intent routing with unknowns | CLINC150 (150 intents + out-of-scope queries) | Many classes, and knowing when it doesn't know |
| C. Receipt fields | CORD v2 receipt text: store, date, total, line items | Extraction: structured output, not just a decision |
| A′. Unseen headlines | 300 Indian headlines from Jul–Oct 2026, after the models' training data, labelled by you | That A's result isn't the models remembering a public dataset |

Licences are checked in step 0. If one doesn't fit, Banking77 or another receipt set takes its place.

**Baselines,** run on the same test items:
- the LLM, one call (Claude Sonnet zero-shot, and the free labeller)
- the LLM, majority of 3
- a semantic cache (nearest neighbour on embeddings)
- standard distillation on the same labels (hashed logistic regression; ModernBERT on Kaggle)
- the program without tiny parts, and the tiny parts without a program
- for A only, the human-trained news specialist (0.88 macro-F1) as an upper reference

**Pass rules, fixed now, before any run.** Each task must meet all six:

1. **Coverage:** the skill answers at least 70% of test inputs on its own.
2. **Quality:** on human labels, skill + fallback scores no more than 1 point below the LLM alone (the lower end of
   a 95% paired-bootstrap interval).
3. **The bound holds:** measured error on the answered test cases is at most the `max_error` chosen on calibration
   (2% vs the labellers' consensus).
4. **Speed:** p50 at most 1 ms without embeddings and at most 10 ms with them, on one CPU core.
5. **Rules:** 0 rule breaks.
6. **Cost:** at most 300 programmer calls of practice; labels at most 3 votes × 2,000 items, batched.

**The POC passes if at least 2 of tasks A, B and C pass, and A′ doesn't contradict A.**

- "Beats the LLM" is claimed only where the interval's lower end is above 0.
- **If it fails,** we publish what we learned and stop this direction. We don't keep tuning until it passes.

## Steps, in order (no dates)

0. **Harness:**
   - datasets and splits by text, with a leak check
   - labellers, batched
   - LLM baselines on the human labels
   - licences
   - the unseen headline set
1. **Sandbox:** program runner, AST allowlist, pack format, tests.
2. **First skill, by hand.** Claude Code acts as the programmer on task C (code is strongest at extraction), with
   the test split sealed: only its hash is visible until the final run. This shows whether the idea works before we
   automate the loop.
3. **Automate:** the evolution loop, the tiny parts, confidence, the Clopper–Pearson threshold, certify.
4. **Verdict:** tasks A, B and A′, judged by the pass rules. Results go in `bench/practice/RESULTS.md`, failures
   included.
5. **If it passed:**
   - `@shishya.skill`, the runtime, corrections → repair, the live audit, demotion
   - release 0.4 (beta)
   - a README rebuilt around one demo
6. **Dogfood:** a compiled headline-triage skill runs in shadow next to kite-signal-bot's watchman. It only reads and
   never changes what the bot does. This step happens only with your OK, because of the bot's own constraints.
7. **Grow:**
   - the proxy with template grouping
   - **PracticeBench:** public tasks, labels and results; anyone can add a task or a system
   - more output types and a JavaScript export

**Research track, not promised:**
- free text (a program plus a small LM)
- agent workflows compiled from Gate traces
- a shared registry of certified skill packs (`shishya pull indian-bank-sms`)

## Open source and money: the decision

- **Everything stays Apache-2.0 and local-first.**
- **Stars don't come automatically. They come from one undeniable demo:**
  1. a 20-second clip: an LLM prompt goes in, a 40-line readable program comes out, then the benchmark table
  2. one surprising, honest number
  3. a free first run: `pip install shishya && shishya practice examples/receipts` with free open labellers and no
     paid key
- **Launch** on Show HN, r/LocalLLaMA, r/MachineLearning, X, LinkedIn and Indian developer groups. PracticeBench
  gives other people a reason to send pull requests.
- **Money later, and only after a signal:** at least 3 teams on real traffic, or 1,000 stars.
  - **What to sell:** hosted practice and a team dashboard (program diffs to approve, live error, drift alerts),
    plus audit reports.
  - **Price:** a share of the measured LLM savings.
  - **Likely first users:** Indian fintech and support teams (SMS parsing, KYC fields, ticket routing), where the
    cost per call matters.
- **Until then:** no cloud service, no sign-ups, no billing code.

## Risks

- **Meaning-heavy tasks** (A) may lean on the tiny parts for most answers. Then the "program" is thin and we are
  close to plain distillation. The pass rules will show it.
- **Programs can memorise.** The size penalty, the calibration split and the "2 examples per table entry" rule are
  there to stop it.
- **The bound covers only traffic like the calibration traffic.** The live audit and demotion handle drift.
- **Label licences** decide which models may teach. The default is permissive open models plus people.
- **Shared quota.** kite-signal-bot uses the same Claude login. Practice runs must stay out of its busy hours
  (00:30–09:45 IST: the night check, the brief, the limit re-run, the AI paper desk) and below about 10% of the
  weekly window, because the bot stops its background AI at 90%. OpenRouter's free calls are shared with the
  watchman's budget too.
- **A large company could copy this.** What protects us is the trust stack (rules, proof, bound, receipts), the open
  benchmark and shipping fast.

## Sources

- Compiled AI: Deterministic Code Generation for LLM-Based Workflow Automation —
  [arXiv 2604.05150](https://arxiv.org/abs/2604.05150), [repo](https://github.com/XY-Corp/CompiledAI)
- ReaComp: Compiling LLM Reasoning into Symbolic Solvers — [arXiv 2605.05485](https://arxiv.org/html/2605.05485v1)
- rote — [PyPI](https://pypi.org/project/rote-cli/), [directory listing](https://enterprisedna.co/directories/open-source/rote)
- Butter (Show HN) — [thread](https://hn.svelte.dev/item/45737948); muscle-mem — [repo](https://github.com/pig-dot-dev/muscle-mem)
- ScriptoriumWS — [arXiv 2502.12366](https://arxiv.org/pdf/2502.12366); DataSculpt —
  [arXiv 2311.00739](https://arxiv.org/abs/2311.00739v1); Alfred — [arXiv 2305.18623](https://arxiv.org/pdf/2305.18623)
- Distilling step-by-step (Hsieh et al., ACL 2023) — [paper](https://arxiv.org/html/2305.02301v2),
  [Google blog](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/)
- Conformal risk control for selective prediction in LMs — [arXiv 2607.24562](https://arxiv.org/pdf/2607.24562)
- CodeEvolve (open evolutionary code search) — [arXiv 2510.14150](https://arxiv.org/html/2510.14150v3)
- System One models and Kev — [overview](https://www.dsebastien.net/system-one-models/),
  [Kev 4B on OpenRouter](https://openrouter.ai/jaredpalmer/kev-4b-20260924)
- Enterprise inference spend (vendor survey, weak) —
  [RapidData 2026](https://www.rapiddata.com/insights/reports/state-of-enterprise-ai-2026.html)
- EU AI Act delay (Digital Omnibus) —
  [Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/)
- Fitts and Posner, *Human Performance* (1967); J. R. Anderson, "Acquisition of cognitive skill",
  *Psychological Review* 89 (1982): knowledge compilation.
