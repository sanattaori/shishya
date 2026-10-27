"""Rules an answer must follow.

Python rules (library use):

    @rule(uses=["verdict"])
    def within_balance(ans, ctx):
        "Leave can be approved only up to the remaining balance."
        return ans["verdict"] != "approve" or ctx["facts"]["days_asked"] <= ctx["facts"]["leave_balance"]

``ans`` holds the decided fields, ``ctx`` is ``{"input", "facts"}``. Return True when the answer is
allowed. With ``uses``, the rule runs as soon as those fields are decided, so a whole branch of answers
is ruled out at once; without it, the rule sees only complete answers. The first line of the docstring
is the rule's message (shown to the Guru and in results).

Rego rules (OPA policies, also accepted by the HTTP server): see ``RegoRules``. They see complete
answers only and are checked in batches with one ``opa eval`` call per batch.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    fn: Callable[[dict, dict], Any]
    uses: tuple[str, ...] | None
    name: str
    message: str
    spec: Any = None                # the JSON definition, for declarative rules

    def __call__(self, ans: dict, ctx: dict) -> bool:
        return bool(self.fn(ans, ctx))

    def fingerprint(self) -> str:
        if self.spec is not None:
            return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()[:16]
        try:
            src = inspect.getsource(self.fn)
        except (OSError, TypeError):
            src = self.name
        return hashlib.sha256(f"{self.name}|{self.uses}|{src}".encode()).hexdigest()[:16]


def rule(fn: Callable | None = None, *, uses: list[str] | tuple[str, ...] | None = None,
         name: str | None = None, message: str | None = None):
    """Decorator (``@rule`` or ``@rule(uses=[...])``) turning a function into a Rule."""
    def wrap(f: Callable) -> Rule:
        if uses is not None and (not uses or not all(isinstance(u, str) for u in uses)):
            raise RuleError("uses must be a non-empty list of field names")
        doc = (inspect.getdoc(f) or "").strip().splitlines()
        return Rule(f, tuple(uses) if uses is not None else None, name or f.__name__,
                    message or (doc[0] if doc else f.__name__))
    return wrap(fn) if fn is not None else wrap


def json_rule(spec: Any, fields: set[str]) -> Rule:
    """A declarative rule: ``{"name", "message"?, "forbid" | "require": <condition>}``.

    Conditions use the language of ``when.py`` and may read ``answer.<field>``, ``facts.*`` and
    ``input``. ``forbid`` rejects an answer when its condition is definitely true; ``require`` rejects
    it unless its condition is definitely true (an unknown fact fails a requirement). No code runs, so
    these rules are safe to accept from anyone, including over HTTP and from an LLM draft.
    """
    from . import when as when_mod

    if not isinstance(spec, dict):
        raise RuleError("a JSON rule must be an object")
    extra = set(spec) - {"name", "message", "forbid", "require"}
    if extra:
        raise RuleError(f"unknown keys in JSON rule: {sorted(extra)}")
    name = spec.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RuleError("a JSON rule needs a 'name'")
    kinds = [k for k in ("forbid", "require") if k in spec]
    if len(kinds) != 1:
        raise RuleError(f"rule {name!r} needs exactly one of 'forbid' or 'require'")
    kind, cond = kinds[0], spec[kinds[0]]
    try:
        when_mod.check(cond)
    except when_mod.ConditionError as e:
        raise RuleError(f"rule {name!r}: {e}") from e
    used = []
    for p in when_mod.paths(cond):
        parts = p.split(".")
        if parts[0] == "answer":
            if len(parts) < 2 or parts[1] not in fields:
                raise RuleError(f"rule {name!r} reads {p!r}, but the schema has no such field")
            if parts[1] not in used:
                used.append(parts[1])
    if not used:
        raise RuleError(f"rule {name!r} never reads the answer (use a field condition in the schema instead)")
    message = spec.get("message") or name
    if not isinstance(message, str):
        raise RuleError(f"rule {name!r}: 'message' must be a string")

    def fn(ans: dict, ctx: dict) -> bool:
        v = when_mod.evaluate(cond, {**ctx, "answer": ans})
        return v is not True if kind == "forbid" else v is True
    return Rule(fn, tuple(used), name, message, spec)


def as_rules(items: Any) -> list[Rule]:
    out = []
    for r in items or ():
        if isinstance(r, Rule):
            out.append(r)
        elif callable(r):
            out.append(rule(r))
        else:
            raise RuleError(f"not a rule: {r!r}")
    return out


# Built-ins a policy may not call: no network, no DNS, no reading the server's environment.
BLOCKED_BUILTINS = {"http.send", "net.lookup_ip_addr", "opa.runtime"}
_CAPS: dict[str, str] = {}


def _capabilities(opa: str) -> str:
    """Path of a capabilities file for ``opa`` without BLOCKED_BUILTINS (made once per binary)."""
    if opa not in _CAPS:
        proc = subprocess.run([opa, "capabilities", "--current"], capture_output=True, text=True, check=False,
                              timeout=30)
        if proc.returncode != 0:
            raise RuleError(f"opa capabilities failed: {proc.stderr.strip()[:300]}")
        caps = json.loads(proc.stdout)
        caps["builtins"] = [b for b in caps["builtins"] if b["name"] not in BLOCKED_BUILTINS]
        fd, path = tempfile.mkstemp(prefix="shishya-opa-caps-", suffix=".json")
        with open(fd, "w") as f:
            json.dump(caps, f)
        _CAPS[opa] = path
    return _CAPS[opa]


class RegoRules:
    """An OPA policy checked against complete answers, many at a time.

    For each candidate the policy sees this ``input``:

        {"answer": {...decided fields...}, "input": ..., "facts": ...}

    and must define ``allow`` (default false) in its package, for example::

        package shell_guard
        import rego.v1
        default allow := false
        allow if input.answer.action != "run"
        allow if { input.answer.action == "run"; not input.answer.destructive }

    It may also define ``why_not`` (a set of strings) to explain a rejection; Shishya passes these
    to the Guru fallback and returns them in the result.

    Policies cannot use ``http.send``, ``net.lookup_ip_addr`` or ``opa.runtime``: a rule check never
    reaches the network or the server's environment, so policies sent over HTTP are safe to run.
    """

    BATCH_PACKAGE = "shishya_batch"

    def __init__(self, policy: str, *, query: str | None = None, opa: str | None = None, timeout: float = 30):
        pkgs = [line.split()[1] for line in policy.splitlines()
                if line.strip().startswith("package ") and len(line.split()) > 1] if isinstance(policy, str) else []
        if not pkgs:
            raise RuleError("a Rego policy must be a string with a 'package' line")
        self.policy = policy
        pkg = pkgs[0]
        self.package = pkg
        self.query = query or f"data.{pkg}.allow"
        if not self.query.startswith("data."):
            raise RuleError("query must start with 'data.'")
        self.reasons_ref = self.query.rsplit(".", 1)[0] + ".why_not"
        self.opa = opa or shutil.which("opa")
        self.timeout = timeout
        self.name = f"rego:{pkg}"

    def fingerprint(self) -> str:
        return hashlib.sha256((self.query + "\n" + self.policy).encode()).hexdigest()[:16]

    def _wrapper(self) -> str:
        return (f"package {self.BATCH_PACKAGE}\n"
                "import rego.v1\n"
                f"allowed contains i if {{\n  some i, c in input.candidates\n  {self.query} with input as c\n}}\n"
                f"reasons[i] := r if {{\n  some i, c in input.candidates\n"
                f"  r := {self.reasons_ref} with input as c\n}}\n")

    def check_batch(self, answers: list[dict], ctx: dict) -> list[tuple[bool, list[str]]]:
        if not answers:
            return []
        if not self.opa:
            raise RuleError("Rego rules need the 'opa' binary on PATH (https://www.openpolicyagent.org/docs#running-opa)")
        cands = [{"answer": a, "input": ctx.get("input"), "facts": ctx.get("facts")} for a in answers]
        with tempfile.TemporaryDirectory(prefix="shishya-rego-") as d:
            Path(d, "policy.rego").write_text(self.policy)
            Path(d, "wrapper.rego").write_text(self._wrapper())
            proc = subprocess.run(
                [self.opa, "eval", "--format", "json", "--stdin-input", "--capabilities", _capabilities(self.opa),
                 "-d", str(Path(d, "policy.rego")), "-d", str(Path(d, "wrapper.rego")),
                 f"data.{self.BATCH_PACKAGE}"],
                input=json.dumps({"candidates": cands}), capture_output=True, text=True, check=False,
                timeout=self.timeout)
        if proc.returncode != 0:
            raise RuleError(f"opa eval failed: {(proc.stderr or proc.stdout).strip()[:500]}")
        try:
            value = json.loads(proc.stdout)["result"][0]["expressions"][0]["value"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuleError(f"unexpected opa output: {proc.stdout[:300]}") from e
        allowed = set(value.get("allowed", []))
        reasons = value.get("reasons", {}) or {}
        out = []
        for i in range(len(answers)):
            r = reasons.get(str(i), reasons.get(i, []))
            out.append((i in allowed, sorted(r) if isinstance(r, list) else []))
        return out
