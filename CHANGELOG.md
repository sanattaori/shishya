# Changelog

## 0.3.0 (2026-10-29)

First release on PyPI: `pip install shishya`.

- **Shishya Gate for Claude Code** (`shishya gate hook|test|status|correct`, plugin `shishya-gate` in `plugins/claude-code`, marketplace file `.claude-plugin/marketplace.json`). A PreToolUse hook that allows, asks or denies a tool call. A shell parser follows `bash -c`, `eval`, `$(...)`, backticks, heredocs, wrappers and `find -exec`, and asks when it cannot. 21 core JSON rules deny catastrophic deletes, secret exfiltration and machine damage, and ask for force-push, protected branches, history loss, databases, production, cloud, publishing, secrets and sudo; projects can add rules but not disable them. What the person approves or rejects becomes learner labels; the Gate allows on its own only when confident and the call is one it can read. About 50 ms per call (`shishya-gate-hook`).
- **Learn from corrections** (`shishya.Learner`, `Shishya(..., learner=DIR)`, `s.correct(d, {...})`, `shishya correct`, `shishya learn status|train|rollback`). Memory: the exact case a person corrected gets that answer next time (one correction moves it to at least 0.8, two in a row to 0.89). Head: a small per-field model on the base model's probabilities and the text, retrained in well under a second after each correction, and used only while it beats the base model on a frozen, human-labelled held-out split (paired bootstrap, at least two different answers). Rules still always win. Rollback is saved as a new version, so a version in a receipt always names one model. A broken learner store falls back to the base model and says so in `reasons`.
- Every receipt has an `id`; with a learner, also the model version and whether memory or the head shaped each field.
- **Rego policies fail closed:** opa missing, a timeout or a broken policy blocks the answer and is listed in `rule_errors` (before, the decision crashed).
- Booleans are not numbers: a Guru answering `1` for a yes/no field is rejected, and JSON rule `eq`, `neq`, `in` and `contains` no longer treat `true` as `1`.
- `import shishya` is about 3x faster (backends load on first use).
- Release workflow: tags `v*` publish to PyPI with trusted publishing.
- **Breaking: rules fail closed.** A JSON `forbid` rule now allows an answer only when its condition is definitely false: if a fact it needs is missing or has the wrong type, the answers it might forbid are blocked (before, they were allowed). `"on_missing": "allow"` restores the old behaviour for one rule. A Python rule that raises blocks the answer and is reported in `Decision.rule_errors` instead of crashing the decision.
- `Decision.missing_facts` lists facts the schema's rules needed but were not sent; `reasons` says whether each rule blocked or stepped aside, and names facts of the wrong type.
- The English rule drafter explains fail-closed rules and no longer prefers `forbid`.
- News baselines: FinBERT, and the LLM asked one headline per request.

## 0.2.0 (2026-10-28)

- **News specialist models** (`shishya:news-base`, `shishya:news-large`): trained on a free Kaggle T4 on human-labelled Indian and US financial headlines (SEntFiN, twitter-financial-news-sentiment; MIT). On held-out Indian headlines: macro-F1 0.88 vs 0.78 for a free big LLM and 0.50 for Kev-0.8B, at 17 ms on a laptop. They download once and are checked with SHA-256.
- `ClassifierBackend` (a local classifier for chosen fields) and `RouterBackend` (each field to its own model); `--route FIELD=MODEL` on the CLI; `shishya[local]` extra.
- **Declarative JSON rules** in `x-shishya.rules` (`forbid` / `require` over `answer.*` and `facts.*`). No code runs, so they're safe over HTTP.
- **English → rules:** `shishya rules SCHEMA "..."` lets a Guru draft JSON rules, validates them, shows answers each rule blocks and allows, and never applies them by itself.
- **Calibration:** `shishya calibrate` fits a temperature per field on your labelled examples and reports coverage and error per confidence level, measured on held-out folds. Use the result with `--calibration`.
- Training recipe and one-command Kaggle runner (`train/news_specialist/run_kaggle.sh`), with leak checks.
- README rebuilt around figures (`docs/make_figures.py`).

## 0.1.0 (2026-10-27)

First release: best-first rule search over decision-model probabilities; Python and Rego rules; schema conditions; System One, chat-logprobs and fake backends; OpenRouter Guru fallback; CLI and HTTP server; rulebench v0. `min_probability` is compared per field (`Decision.confidence`, the weakest field), not with the joint probability. Claude Code Guru (`claude -p` with your own login); Guru answers are marked `trainable: false`.
