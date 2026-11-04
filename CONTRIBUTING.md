# Contributing

Issues and pull requests are welcome. The [open problems](docs/RESEARCH.md#open-problems) are a good place to start.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" torch numpy huggingface_hub
.venv/bin/python -m pytest -q                # the library
.venv/bin/python -m pytest brain/tests -q    # the research code
.venv/bin/ruff check .
```

## If you run an experiment

We hold results to one rule: **write the plan, the pass rules and your prediction before the run, and don't change them
after.**

- Make choices on dev names. Use test names once, for the final result.
- Report failures as plainly as passes, and include a baseline run on the same questions.
- Add the small raw outputs to `brain/results/`.

## Writing style

Use short, plain sentences. If a number could mislead, say what it doesn't show.
