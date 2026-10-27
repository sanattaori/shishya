"""Command line: ``shishya decide``, ``shishya serve``, ``shishya version``."""
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
    guru = None
    if a.guru == "claude":
        from .guru import ClaudeCodeGuru
        guru = ClaudeCodeGuru()
    elif a.guru == "openrouter":
        from .guru import OpenRouterGuru
        guru = OpenRouterGuru()
    engine = Shishya(_backend(a.backend), rules=_load_rules(a.rules) if a.rules else (), rego=rego, guru=guru,
                     min_probability=a.min_probability)
    d = engine.decide(text, schema, facts=facts)
    if a.json:
        print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
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
        t = d.receipt.get("timings_ms", {})
        print("timings: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))
    return 0 if d.status in ("ok", "guru") else 2


def _serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import create_app
        app = create_app(_backend(a.backend))
    except ImportError:
        raise ValueError("the server needs extra packages: pip install 'shishya[server]'") from None
    uvicorn.run(app, host=a.host, port=a.port)
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
    d.add_argument("--json", action="store_true", help="print the full result as JSON")
    d.set_defaults(fn=_decide)
    s = sub.add_parser("serve", help="run the HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8700)
    s.add_argument("--backend", default=default_backend)
    s.set_defaults(fn=_serve)
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
