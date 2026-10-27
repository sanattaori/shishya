# Shishya

[![CI](https://github.com/sanattaori/shishya/actions/workflows/ci.yml/badge.svg)](https://github.com/sanattaori/shishya/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Give any AI model your rules. Shishya returns the most likely answer that follows them, in milliseconds, or tells you it can't.**

On our 60-task benchmark, a small local decision model broke a rule on **38%** of tasks: it was ready to run `rm -rf /`, approve leave beyond the balance, and buy stock while the market guard was on. With Shishya on top, the same model call broke **0%**, and correct answers went from 42% to **73%**, for **11 ms** of extra work on a laptop. ([results](bench/RESULTS.md))

Small decision models (Kev, jev-style and other Jev-compatible servers) are fast and give a probability for every option. But they don't know *your* rules: leave balances, position limits, which shell commands an agent may run on its own. Prompting the rules in doesn't make them reliable, and the usual alternative, asking an LLM for JSON and retrying when a validator fails, costs a full LLM call per retry.

Shishya takes the model's probabilities and searches, best first, for the most likely answer that **passes every rule**. Rules are plain Python functions or OPA/Rego policies. The answer it returns can't break a rule, because each one is checked before it comes back.

```
model says:   verdict = approve (70%)   hr_review (20%)   reject (10%)
rule:         "Leave can be approved only up to the remaining balance"   (5 days asked, 3 left)
Shishya:      verdict = hr_review       one rule check, 0.03 ms of search
```

- **Exact:** the search returns exactly the most likely allowed answer (checked against brute force on 300 random problems in the tests).
- **Fast:** rules that name their fields cut whole branches at once. One model call, then microseconds of search; Rego policies are checked in batches.
- **Honest:** every result says how much of the model's probability the rules removed, which rules fired, and the model's own top answer, plus a receipt (schema digest, rule fingerprints, model, timings).
- **Any model:** works with any System One–compatible decision server, with OpenAI-compatible chat endpoints that return log-probabilities (including free OpenRouter models), or with a fake backend for tests.
- **A Guru when stuck:** if any field of the best allowed answer is below your confidence bar, Shishya can ask a bigger LLM (the *Guru*), tell it which rules were broken, and check its answer against the same rules.

> *Shishya* (शिष्य) is Sanskrit for "student", one who learns from a *guru*.

## Quickstart (30 seconds, no model needed)

```bash
pip install "git+https://github.com/sanattaori/shishya"
git clone https://github.com/sanattaori/shishya && cd shishya

shishya decide examples/leave_request/schema.json \
  --input "Need 5 days off next week for my sister's wedding" \
  --facts examples/leave_request/facts.json \
  --rules examples/leave_request/rules.py \
  --backend fake
```

```
status: ok
answer: {"verdict": "hr_review", "urgent": true, "probation_note": null}
model's top answer: {"verdict": "approve", "urgent": true, "probation_note": null}
broken rule: within_balance (rejected 1)
```

`--backend fake` uses deterministic made-up probabilities, so you can see the mechanism. For real answers, use a real model (below).

## With a real model

Any server that speaks the System One decision API works. For example, [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0) runs on a laptop:

```bash
# in the Kev repo
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009   # or kev-4b
```

```bash
shishya decide examples/shell_guard/schema.json \
  --input "The agent wants to run: rm -rf build/ data/" \
  --facts examples/shell_guard/facts.json \
  --rego examples/shell_guard/policy.rego \
  --backend http://127.0.0.1:8009
```

Or a hosted free model that returns log-probabilities (needs `OPENROUTER_API_KEY`):

```bash
shishya decide ... --backend openrouter        # liquid/lfm-2.5-2.6b:free by default
```

Free hosted models are often rate-limited, and few return log-probabilities, so treat this backend as a way to try Shishya, not as a production setup.

## In Python

```python
from shishya import Shishya, rule
from shishya.backends import SystemOneBackend

schema = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string", "enum": ["approve", "reject", "hr_review"],
            "description": "What should happen to this leave request?",
            "x-shishya": {"options": {"approve": "A clear, reasonable request",
                                      "reject": "No acceptable reason",
                                      "hr_review": "Unusual or sensitive; a person decides"}},
        },
        "urgent": {"type": "boolean", "description": "Is this an emergency?"},
    },
}

@rule(uses=["verdict"])
def within_balance(ans, ctx):
    "Leave can be approved only up to the remaining balance."
    f = ctx["facts"]
    return ans["verdict"] != "approve" or f["days_asked"] <= f["leave_balance"]

s = Shishya(SystemOneBackend("http://127.0.0.1:8009"), rules=[within_balance])
d = s.decide("Need 5 days off next week for my sister's wedding", schema,
             facts={"days_asked": 5, "leave_balance": 3})

d.status        # "ok" | "guru" | "unsure" | "none"
d.answer        # {"verdict": "hr_review", "urgent": False}
d.probability   # the model's joint probability for this answer
d.confidence    # the weakest field's probability (what min_probability is compared with)
d.top           # the model's own top answer, before rules
d.broken_rules  # {"within_balance": 1}
d.receipt       # schema digest, rule fingerprints, backend, timings
```

## How it works

```
 input + facts ─┐
                ▼
   1. conditions   x-shishya.when / option_when are plain code: they remove fields and options
                   the facts rule out before the model sees them
   2. model        ONE call: a probability for every allowed option of every field
   3. search       A* over the fields. Priority = probability so far × best possible rest,
                   so complete answers come out most-likely-first.
                   Python rules with `uses=[...]` run as soon as those fields are set and cut the
                   whole branch; Rego policies check complete answers in batches.
   4. result       ok      the most likely answer that passes every rule
                   unsure  an allowed answer exists, but some field is below min_probability
                           (returned as suggestion)
                   guru    a bigger LLM answered, told which rules failed, and its answer passed the rules
                   none    no allowed answer within the search budget
```

Fields are scored independently given the input (v0.1). That's what makes one model call enough, and it's also the main limitation; see below.

## Rules

**Python**, for library use. Return `True` when the answer is allowed. The first docstring line is the message.

```python
@rule(uses=["action", "tell_owner"])
def owner_hears_about_sells(ans, ctx):
    "Any sell must be told to the owner."
    return ans["action"] != "trim" or ans["tell_owner"]
```

**Rego (OPA)**, for policies kept outside code, and the only rules the HTTP server accepts. The policy sees `input.answer`, `input.input` and `input.facts`, must define `allow`, and may explain itself with `why_not`:

```rego
package shell_guard
import rego.v1

default allow := false
allow if input.answer.action != "run"
allow if {
    input.answer.action == "run"
    not input.answer.destructive
    not input.answer.network
}
why_not contains "destructive commands always need a person" if {
    input.answer.action == "run"
    input.answer.destructive
}
```

Rego needs the [`opa`](https://www.openpolicyagent.org/docs#running-opa) binary on `PATH`. Policies cannot call `http.send`, `net.lookup_ip_addr` or `opa.runtime`, so a rule check never reaches the network or the server's environment.

**Conditions** (in the schema, evaluated without the model):

```json
"x-shishya": {
  "when": {"path": "facts.on_probation", "eq": true},
  "option_when": {"trim": {"path": "facts.owned_qty", "gt": 0}}
}
```

Operators: `eq neq gt gte lt lte in contains exists all any not`. A missing fact is *unknown* and never makes a field or option eligible.

## The Guru

When some field of the best allowed answer is below `min_probability`, Shishya can ask a bigger model, the *Guru*. The Guru is told which rules the earlier answers broke, and its answer must pass the same rules before it's returned.

```bash
shishya decide ... --min-probability 0.9 --guru claude       # Claude through your own Claude Code login (claude -p)
shishya decide ... --min-probability 0.9 --guru openrouter   # any OpenRouter model (OPENROUTER_API_KEY)
```

```python
from shishya.guru import ClaudeCodeGuru, OpenRouterGuru
Shishya(backend, rules=[...], guru=ClaudeCodeGuru(model="sonnet"), min_probability=0.9)
```

`ClaudeCodeGuru` runs `claude -p` with no tools, no MCP servers and no saved session, in an empty temporary folder, and checks the reply against a JSON Schema built from your fields.

Guru answers come back with `trainable: False`. Shishya's learning loop (v0.3) will never train on them; provider terms often forbid training on their outputs.

## HTTP server

```bash
pip install "shishya[server] @ git+https://github.com/sanattaori/shishya"
shishya serve --backend http://127.0.0.1:8009 --port 8700
```

```bash
curl -s localhost:8700/v1/decide -H 'content-type: application/json' -d '{
  "input": "The agent wants to run: git push --force origin main",
  "schema": {...}, "facts": {...}, "rego": "package shell_guard ..."
}'
```

## Benchmark

`bench/` has **rulebench v0**: 60 tasks across three scenarios (leave requests, stock alerts, shell commands). Each task has a correct answer, and the model is never shown the rules. Kev-0.8B ran locally on an M2 MacBook (16 GB); the LLM is free Nemotron on OpenRouter.

| System | Correct | Breaks a rule | p50 | p95 |
|---|---:|---:|---:|---:|
| Kev-0.8B alone | 42% | 38% | 184 ms | 250 ms |
| **Kev-0.8B + Shishya** | **73%** | **0%** | **195 ms** | **252 ms** |
| LLM writes JSON, retries up to 3× when a rule fails | 83% | 0% | 5,048 ms | 42,685 ms |

The big LLM judges the *text* better (mainly stock news), but it's 26× slower at the median and took up to 43 s. Shishya's job is the rules: it removed every rule break of the small model with no extra model call. Full breakdown, the Guru runs and what we learned: [`bench/RESULTS.md`](bench/RESULTS.md).

```bash
python bench/make_tasks.py > bench/tasks.jsonl
python bench/run.py --backend http://127.0.0.1:8009 --llm 60
```

The authors wrote this benchmark, so it favours the cases rules are good at. Read it as a demonstration of the mechanism, not as a measure of model quality.

## Compared with

- **Validate and retry** (e.g. Instructor, or a JSON-mode LLM in a loop): works with any LLM. Each failed rule costs a full LLM call, and nothing guarantees the retry finds the *most likely* allowed answer. Shishya makes one model call and searches.
- **Decision servers on their own** (Kev, jev-style, other Jev-compatible servers): fast, and they give probabilities, but they don't know your rules. Shishya sits on top of them.
- **Policy engines** (OPA, Cedar): check an answer someone already chose. Shishya uses them *inside* the choice.

## Limits (v0.1)

- Only `boolean` and string `enum` fields. No numbers, free text or quotes from the input yet.
- Fields are scored independently given the input, so the model's view of one field doesn't change after another is chosen. Rules can couple fields; the model's probabilities don't.
- `probability` is the model's own number. Most decision models are not well calibrated on new tasks, so measure on your own labelled examples before trusting a threshold.
- If the model is wrong about the facts in the text, rules can't fix that. They only stop answers your policy forbids.

## Roadmap

- **v0.2:** a small stock-news specialist trained on free Kaggle GPUs (labels from two open-weight teachers that must agree, tested on human-labelled data), numbers and quoted spans from the input (checked against the source), JSON Schema `if/then`, re-scoring a field given earlier choices.
- **v0.3:** *Shishya learns.* Guru answers and your corrections train a small local model that takes over repeat cases. It trains on free Kaggle GPUs and is only used if it beats the current one on held-out data.
- MCP server, an agent tool-call guard, a TypeScript client.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/ruff check .
```

## Credits

Shishya calls decision models over HTTP and includes none of their code or weights. Thanks to the authors of [Kev](https://github.com/jaredpalmer/kev) and [jev-style](https://github.com/lawrence3699/jev-style) (both Apache-2.0) for open decision models, and to [Open Policy Agent](https://www.openpolicyagent.org/).

## License

Apache-2.0
