# Changelog

## Unreleased

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
