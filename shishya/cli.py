"""Command line: ``shishya decide``, ``serve``, ``calibrate``, ``rules``, ``correct``, ``learn`` and ``version``."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_BACKEND = "http://127.0.0.1:8009"


def _backend(spec: str):
    from .backends import ChatLogprobsBackend, FakeBackend, SystemOneBackend
    if spec == "fake":
        return FakeBackend()
    if spec == "openrouter":
        return ChatLogprobsBackend()
    if spec.startswith("http"):
        return SystemOneBackend(spec)
    raise ValueError(f"unknown backend {spec!r}: use a URL, 'fake' or 'openrouter'")


def _routed(backend, routes: list[str] | None):
    """--route FIELD=MODEL (repeatable): that field is answered by a local classifier (e.g. shishya:news-base).

    The field's options must be the classifier's labels (for shishya:news-*: bearish, bullish, neutral)."""
    if not routes:
        return backend
    from .backends import ClassifierBackend, RouterBackend
    table = {}
    for spec in routes:
        field, sep, model = spec.partition("=")
        if not sep or not field or not model:
            raise ValueError(f"--route needs FIELD=MODEL (got {spec!r})")
        table[field] = ClassifierBackend(model, fields={field: {lab: lab for lab in _labels_of(model)}})
    return RouterBackend(default=backend, routes=table)


def _labels_of(model: str) -> list[str]:
    from .models import MODELS
    if model.startswith("shishya:") and model[8:] in MODELS:
        return MODELS[model[8:]]["labels"]
    import json as _json
    cfg = _json.loads((Path(model) / "config.json").read_text())
    return [v for _, v in sorted(cfg["id2label"].items(), key=lambda kv: int(kv[0]))]


def _calibrated(backend, path: str | None):
    if not path:
        return backend
    from .calibrate import CalibratedBackend, load
    return CalibratedBackend(backend, load(path))


def _guru(name: str | None):
    if name == "claude":
        from .guru import ClaudeCodeGuru
        return ClaudeCodeGuru()
    if name == "openrouter":
        from .guru import OpenRouterGuru
        return OpenRouterGuru()
    return None


def _load_rules(path: str) -> list:
    from .rules import Rule
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"shishya_user_rules_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load rules from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = [v for v in vars(mod).values() if isinstance(v, Rule)]
    if not found:
        raise ValueError(f"no shishya rules found in {path}")
    return found


def _json_arg(value: str) -> Any:
    """A JSON string, or the path of a JSON file (optionally written ``@path``)."""
    if value.lstrip().startswith(("{", "[")):
        return json.loads(value)
    return json.loads(Path(value.removeprefix("@")).read_text())


def _decide(a: argparse.Namespace) -> int:
    from .engine import Shishya
    schema = json.loads(Path(a.schema).read_text())
    text = Path(a.input[1:]).read_text() if a.input.startswith("@") else a.input
    facts = _json_arg(a.facts) if a.facts else None
    rego = Path(a.rego).read_text() if a.rego else None
    guru = _guru(a.guru)
    backend = _calibrated(_routed(_backend(a.backend), a.route), a.calibration)
    engine = Shishya(backend, rules=_load_rules(a.rules) if a.rules else (), rego=rego, guru=guru,
                     min_probability=a.min_probability, learner=a.learner)
    d = engine.decide(text, schema, facts=facts)
    if a.json:
        print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"decision id: {d.receipt['id']}")
        print(f"status: {d.status}")
        shown = d.answer or d.suggestion
        if shown is not None:
            print(f"{'answer' if d.answer else 'suggestion'}: {json.dumps(shown, ensure_ascii=False)}")
        if d.status in ("ok", "unsure"):
            print(f"probability: {d.probability:.3f}")
        if d.top is not None and d.top != shown:
            print(f"model's top answer: {json.dumps(d.top, ensure_ascii=False)}")
        for name, n in d.broken_rules.items():
            print(f"broken rule: {name} (rejected {n})")
        for why in d.reasons:
            print(f"note: {why}")
        for err in d.rule_errors:
            print(f"rule error (blocked): {err}")
        t = d.receipt.get("timings_ms", {})
        print("timings: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))
    return 0 if d.status in ("ok", "guru") else 2


def _serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import create_app
        app = create_app(_calibrated(_routed(_backend(a.backend), a.route), a.calibration))
    except ImportError:
        raise ValueError("the server needs extra packages: pip install 'shishya[server]'") from None
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:4.1f}%"


def _calibrate(a: argparse.Namespace) -> int:
    from . import calibrate as cal
    from .schema import digest
    schema = json.loads(Path(a.schema).read_text())
    backend = _backend(a.backend)
    collected = cal.collect(backend, schema, cal.load_labels(a.labels))
    if not collected:
        raise ValueError("no usable labels: every label was for a field the model would not be asked about, "
                         "or not an allowed option")
    temps = cal.fit(collected)
    rep = cal.report(collected, temps)
    cal.save(a.out, temps, {"schema_digest": digest(schema), "backend": backend.name,
                            "model": getattr(backend, "model", None), "examples": len(collected)})
    print(f"{rep['n_examples']} examples used, {rep['skipped_labels']} labels skipped. "
          "Calibrated numbers come from 2-fold cross-fitting (fitted on one half, measured on the other).")
    if rep["n_examples"] < 50:
        print("warning: fewer than 50 examples, so these numbers are rough")
    print(f"\n{'field':<24}{'n':>5}{'T':>7}{'accuracy':>10}   ECE before -> after")
    for name, r in rep["fields"].items():
        print(f"{name:<24}{r['n']:>5}{temps[name]:>7.2f}{_pct(r['accuracy']):>10}   "
              f"{_pct(r['ece_before'])} -> {_pct(r['ece_after'])}")
    print("\nAll fields (every labelled field must clear the bar):")
    print(f"{'min probability':<18}{'automated':>10}{'error':>9}")
    for r in rep["all_fields"]["thresholds"]:
        print(f"{r['threshold']:<18.2f}{_pct(r['coverage']):>10}{_pct(r['error']):>9}")
    print(f"\nwrote {a.out}; use it with: shishya decide ... --calibration {a.out}")
    return 0


def _print_train(rep: dict | None) -> None:
    if not rep:
        return
    print(f"model version {rep['version']}" + (" (changed)" if rep["changed"] else " (unchanged)")
          + f", trained in {rep['train_ms']:.0f} ms")
    for name, r in rep["fields"].items():
        state = "head used" if r["promoted"] else "base model used"
        print(f"  {name}: {state}; {r['n_train']} train / {r['n_test']} test labels; {r['why']}")


def _correct(a: argparse.Namespace) -> int:
    from .learn import Learner
    res = Learner(a.dir).correct(a.decision_id, _json_arg(a.answer), source=a.source, train=not a.no_train)
    print("recorded: " + (", ".join(res["recorded"]) or "nothing"))
    for name, why in res["skipped"].items():
        print(f"skipped {name}: {why}")
    _print_train(res["train"])
    return 0 if res["recorded"] else 1


def _learn(a: argparse.Namespace) -> int:
    from .learn import Learner
    learner = Learner(a.dir)
    if a.action == "train":
        _print_train(learner.train())
    elif a.action == "rollback":
        print(f"now using model version {learner.rollback()}")
    else:
        st = learner.status()
        print(f"model version {st['version']}, {st['decisions']} decisions logged")
        print(f"{'field':<24}{'train':>7}{'test':>6}{'memory':>8}  head")
        for name, f in st["fields"].items():
            print(f"{name:<24}{f['train']:>7}{f['test']:>6}{f['memory']:>8}  {'used' if f['promoted'] else 'not used'}")
        if st["last_report"]:
            print("last train:")
            _print_train(st["last_report"])
    return 0


def _rules(a: argparse.Namespace) -> int:
    from .draft import draft_rules, examples
    schema = json.loads(Path(a.schema).read_text())
    facts = _json_arg(a.facts) if a.facts else None
    drafted = draft_rules(_guru(a.guru), schema, a.english, facts)
    print(json.dumps({"rules": drafted}, indent=2, ensure_ascii=False))
    rows = examples(schema, drafted, facts or {})
    print(f"\nExamples (facts: {json.dumps(facts or {}, ensure_ascii=False)}):")
    for r in rows:
        mark = "BLOCK" if r["blocked_by"] else "allow"
        why = f"  <- {', '.join(r['blocked_by'])}" if r["blocked_by"] else ""
        print(f"  {mark}  {json.dumps(r['answer'], ensure_ascii=False)}{why}")
    if not any(r["blocked_by"] for r in rows):
        print("  (no answer is blocked with these facts)")
    if a.out:
        Path(a.out).write_text(json.dumps({"rules": drafted}, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")
    print("\nReview these rules, then add them to the schema under x-shishya.rules. Nothing was applied.")
    return 0


def main(argv: list[str] | None = None) -> int:
    default_backend = os.environ.get("SHISHYA_BACKEND", DEFAULT_BACKEND)
    ap = argparse.ArgumentParser(prog="shishya", description="The most likely answer that follows your rules.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="decide one input")
    d.add_argument("schema", help="schema JSON file")
    d.add_argument("--input", required=True, help="text, or @file")
    d.add_argument("--facts", help="facts as a JSON file path or a JSON string")
    d.add_argument("--rules", help="Python file with shishya rules")
    d.add_argument("--rego", help="Rego policy file")
    d.add_argument("--backend", default=default_backend, help="URL, 'fake' or 'openrouter'")
    d.add_argument("--guru", nargs="?", const="claude", choices=["claude", "openrouter"],
                   help="when no allowed answer is likely enough, ask a Guru: 'claude' (your Claude Code login, "
                        "the default) or 'openrouter' (OPENROUTER_API_KEY)")
    d.add_argument("--min-probability", type=float, default=0.0)
    d.add_argument("--calibration", help="calibration file made by 'shishya calibrate'")
    d.add_argument("--route", action="append", metavar="FIELD=MODEL",
                   help="answer FIELD with a local classifier, e.g. news_impact=shishya:news-base (repeatable)")
    d.add_argument("--learner", metavar="DIR", help="learn from corrections kept in DIR (created if missing)")
    d.add_argument("--json", action="store_true", help="print the full result as JSON")
    d.set_defaults(fn=_decide)
    s = sub.add_parser("serve", help="run the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8700)
    s.add_argument("--backend", default=default_backend)
    s.add_argument("--calibration", help="calibration file made by 'shishya calibrate'")
    s.add_argument("--route", action="append", metavar="FIELD=MODEL",
                   help="answer FIELD with a local classifier, e.g. news_impact=shishya:news-base (repeatable)")
    s.set_defaults(fn=_serve)
    c = sub.add_parser("calibrate", help="fit a temperature per field on labelled examples")
    c.add_argument("schema", help="schema JSON file")
    c.add_argument("labels", help="JSONL file: {\"input\": ..., \"facts\": ..., \"answer\": {field: value}}")
    c.add_argument("--backend", default=default_backend, help="URL, 'fake' or 'openrouter'")
    c.add_argument("--out", default="calibration.json", help="where to write the temperatures")
    c.set_defaults(fn=_calibrate)
    r = sub.add_parser("rules", help="draft rules from plain English (nothing is applied)")
    r.add_argument("schema", help="schema JSON file")
    r.add_argument("english", help="the rules, in plain English")
    r.add_argument("--facts", help="example facts as a JSON file path or a JSON string")
    r.add_argument("--guru", choices=["claude", "openrouter"], default="claude")
    r.add_argument("--out", help="write {\"rules\": [...]} to this file (the schema is never edited)")
    r.set_defaults(fn=_rules)
    co = sub.add_parser("correct", help="give the right answer for a decision, so the learner learns from it")
    co.add_argument("dir", help="learner directory (as in decide --learner)")
    co.add_argument("decision_id", help="the 'decision id' printed by decide")
    co.add_argument("answer", help="JSON file path or JSON string: {\"field\": value}")
    co.add_argument("--source", default="human", help="who says so: 'human' or 'approval' may form the test split")
    co.add_argument("--no-train", action="store_true", help="only record the labels")
    co.set_defaults(fn=_correct)
    le = sub.add_parser("learn", help="inspect, retrain or roll back a learner")
    le.add_argument("action", choices=["status", "train", "rollback"])
    le.add_argument("dir", help="learner directory")
    le.set_defaults(fn=_learn)
    v = sub.add_parser("version", help="print the version")
    v.set_defaults(fn=lambda a: print(__version__) or 0)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (OSError, ValueError, RuntimeError) as e:    # includes SchemaError, RuleError, BackendError
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
