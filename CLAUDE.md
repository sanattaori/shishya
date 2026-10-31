# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project venv is `.venv/` (system Python has no pytest/ruff). Install with `pip install -e ".[dev]"`.

```bash
.venv/bin/python -m pytest -q                                   # full suite (~5 s)
.venv/bin/python -m pytest tests/test_search.py::test_name -q   # one test
.venv/bin/ruff check .                                          # lint (CI runs this; line length 120)
```

Rego tests are skipped unless the `opa` binary is on PATH (CI installs it). Extras: `[server]` (FastAPI),
`[local]` (torch/transformers, needed for `ClassifierBackend` / `--route`), `[dev]`.

Try the pipeline without a model server (`--backend fake` gives deterministic hash-based probabilities):

```bash
shishya decide examples/leave_request/schema.json --backend fake \
  --input "Need 5 days off next week" --facts examples/leave_request/facts.json --rules examples/leave_request/rules.py
```

`bench/run.py` (rulebench, needs a System One server such as Kev on :8009) and `train/news_specialist/` (Kaggle
training) are outside the package; `bench/` and `train/` have relaxed ruff rules.

`brain/` is research outside the package (needs torch): Shishya Brain, BDH plus lasting synapses written by a local
surprise- and reward-weighted delta rule, and the life test against GPT and plain BDH. The plan and its pass rules,
fixed before any run, are in `docs/PLAN.md`; don't change them after seeing results. `brain/third_party/bdh.py` is
Pathway's MIT code, vendored unchanged and excluded from ruff; `brain/tests` (run separately:
`.venv/bin/python -m pytest brain/tests -q`) checks that our BDH matches it exactly. Full runs go to Kaggle
(`brain/kaggle/run.sh`); `brain/out/` is gitignored. The parked compile-to-program plan is in `docs/parked/`.

## Architecture

Shishya returns the most likely answer that breaks no rule. `Shishya.decide` in `shishya/engine.py` runs four steps:

1. **Schema → fields** (`schema.py`): a JSON Schema subset, top-level object of `boolean` (kind `"noul"`) or string
   `enum` (kind `"choice"`) properties. The `x-shishya` blocks carry option text, `when` / `option_when` conditions,
   field `order` (search order) and declarative `rules`. Compiled schemas are cached by content digest; key order
   is part of the digest on purpose.
2. **Conditions** (`when.py`): a small JSON condition language evaluated by plain code, never by the model.
   Evaluation is three-valued (True/False/None = unknown). Missing paths are unknown, `not unknown` is unknown, and
   gt/gte/lt/lte never coerce types. Fields whose `when` fails are fixed to `null`; a field left with one allowed
   option is fixed to it without asking the model.
3. **One backend call** (`backends/`): `score(text, fields, options)` returns probabilities per field over the
   still-allowed options only. The text is `render(input, facts)`; `ClassifierBackend` strips the facts block via
   `FACTS_HEADER`. `RouterBackend` sends chosen fields to other backends. `CalibratedBackend` (`calibrate.py`)
   wraps any backend with per-field temperatures.
4. **A\* search** (`search.py`): fields are treated as independent, so `best_rest` is an exact bound and complete
   answers pop out in true best-first order. Rules with `uses` are checked at the depth of their deepest searched
   field (pruning whole branches). Rules without `uses` run on complete answers. Batch checkers (`RegoRules`, one
   `opa eval` per batch) run on complete answers in order, so the result stays exact.

Then `min_probability` is compared with `Decision.confidence` (the weakest field's probability, not the joint
probability). Below it the result is `unsure`. With a Guru (`guru.py`: `ClaudeCodeGuru` via `claude -p`, or
`OpenRouterGuru`) the broken rules become feedback, and the Guru's answer must pass the same rules and checkers.
Guru answers are marked `trainable: false`.

### Invariants to preserve

- **Rules fail closed.** JSON rules (`rules.json_rule`): `forbid` allows only when its condition is definitely
  False, and `require` only when it is definitely True. `on_missing: "allow"` is the only opt-out (forbid only).
  Python rules are wrapped by `engine._fail_closed`: an exception blocks the answer and lands in
  `Decision.rule_errors`. `engine.fact_notes` explains missing or wrong-typed facts in `reasons` / `missing_facts`.
- **No user code over HTTP.** `server.py` accepts only JSON rules (inside the schema) and Rego. `RegoRules` runs
  OPA with a capabilities file that removes `http.send`, `net.lookup_ip_addr` and `opa.runtime`.
- **The English rule drafter (`draft.py`) only drafts.** LLM output is validated by `json_rule`, shown with
  example allow/block rows, and never written into the schema.
- **Specialist models** (`models.py`) download from GitHub releases and are SHA-256 verified before extraction.
  Registering a new one means adding its checksum and labels to `MODELS`. `--route FIELD=shishya:<name>` requires
  the field's enum values to equal the classifier's labels.
- `Decision.receipt` (schema digest, rule fingerprints, backend/model, timings) is the audit trail. Keep it
  complete when adding rule types or backends.

### Writing style

Docstrings, error messages and the README use short, plain sentences ("the answers it might forbid are blocked").
Match that. User-visible behaviour changes go in `CHANGELOG.md` under "Unreleased". README benchmark numbers come
from `bench/RESULTS.md` and `train/news_specialist/RESULTS.md`; the figures are generated by `docs/make_figures.py`.
