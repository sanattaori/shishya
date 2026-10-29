"""The Gate's decision for one tool call: rules first, then what it has learned from you.

    gate_decide("Bash", {"command": "git push --force origin main"}, cwd="/work/app")
    # {"decision": "ask", "reason": "Force-pushing rewrites shared history; a person decides.", "forced": True, ...}

The decision field has three options: allow, ask, deny. The core rules (``CORE_RULES``, plain JSON you can read)
forbid "allow" for risky calls and forbid everything but "deny" for catastrophic ones. Among what the rules leave,
a prior (or a decision model, if configured) and the learner pick. The Gate then answers:

* deny / ask when a rule forced it (``forced: True``);
* allow only when the learner is confident (``min_probability``, default 0.95) and the call is one the Gate can
  vouch for (it parsed, it runs no inline code or unknown script);
* otherwise no opinion (``None``): Claude Code's own permission flow decides, as if the Gate were not there.

Config (all optional):

* ``~/.shishya/gate/config.json``: ``{"backend": "prior" | URL, "min_probability": 0.95, "rules": [...],
  "disable": ["rule_name", ...]}``
* ``<project>/.shishya/gate.json``: ``{"rules": [...]}``. A project can only ADD rules: it cannot disable core
  rules or change the bar, so a cloned repository cannot switch the Gate off.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .facts import FACT_KEYS, project_root, tool_facts

DEFAULT_MIN_PROBABILITY = 0.95


def _forbid(name: str, message: str, fact: str, answers: tuple[str, ...] = ("allow",)) -> dict:
    when = {"path": "answer.decision", "eq": answers[0]} if len(answers) == 1 else \
        {"path": "answer.decision", "in": list(answers)}
    return {"name": name, "message": message, "forbid": {"all": [{"path": f"facts.{fact}", "eq": True}, when]}}


DENY = ("allow", "ask")
CORE_RULES = [
    _forbid("no_catastrophic_delete", "Deleting a system folder, your home folder or the whole project is blocked.",
            "delete_catastrophic", DENY),
    _forbid("no_secret_exfiltration", "Sending secret files over the network is blocked.", "secret_exfil", DENY),
    _forbid("no_system_damage", "Commands that can wreck the machine (disks, shutdown, kill everything) are blocked.",
            "system_danger", DENY),
    _forbid("recursive_delete_asks", "Recursive deletes need a person.", "delete_recursive"),
    _forbid("delete_outside_project_asks", "Deleting outside the project needs a person.", "delete_outside_project"),
    _forbid("force_push_asks", "Force-pushing rewrites shared history; a person decides.", "git_force_push"),
    _forbid("protected_branch_push_asks", "Pushing straight to a protected branch needs a person.",
            "git_push_protected"),
    _forbid("remote_branch_delete_asks", "Deleting remote branches needs a person.", "git_delete_remote"),
    _forbid("discard_work_asks", "Commands that throw away uncommitted work or history need a person.",
            "git_discard"),
    _forbid("database_destroy_asks", "Dropping, truncating or wiping a database needs a person.", "db_destructive"),
    _forbid("production_asks", "Anything that changes production needs a person.", "mentions_production"),
    _forbid("cloud_destroy_asks", "Destroying cloud resources needs a person.", "cloud_destructive"),
    _forbid("publish_asks", "Publishing packages or releases needs a person.", "publishes"),
    _forbid("remote_code_asks", "Running code downloaded from the internet needs a person.", "remote_code"),
    _forbid("secret_read_asks", "Reading secret files (keys, .env, credentials) needs a person.", "reads_secret"),
    _forbid("sudo_asks", "Commands run as another user (sudo) need a person.", "sudo"),
    _forbid("outside_write_asks", "Writing outside the project needs a person.", "writes_outside_project"),
    _forbid("agent_settings_asks", "Changing the agent's own settings, hooks or the Gate needs a person.",
            "writes_agent_settings"),
    _forbid("shell_config_asks", "Changing shell start-up files needs a person.", "writes_shell_config"),
    _forbid("git_internals_asks", "Writing inside .git needs a person.", "writes_git_internals"),
    _forbid("unparsed_asks", "The Gate could not fully read this command, so a person decides.", "unparsed"),
]

SCHEMA: dict = {
    "type": "object",
    "description": "Should a coding agent's tool call run without asking the person?",
    "properties": {
        "decision": {
            "type": "string", "enum": ["allow", "ask", "deny"],
            "description": "Should this tool call run without asking the person?",
            "x-shishya": {"options": {"allow": "Safe and routine for this project: run it",
                                      "ask": "Let the person decide",
                                      "deny": "Never run this"}},
        },
    },
}


class PriorBackend:
    """No model: a fixed prior. Read-only calls lean towards allow, the rest towards ask. The learner does the rest."""

    name = "prior"
    model = "prior-1"

    def __init__(self) -> None:
        self.read_only = False

    def score(self, text: str, fields: list, options: dict) -> dict:
        allow = 0.6 if self.read_only else 0.4
        p = {"allow": allow, "ask": 1 - allow - 0.01, "deny": 0.01}
        return {f.name: {o: p[o] for o in options[f.name]} for f in fields}


def default_dir() -> Path:
    return Path(os.environ.get("SHISHYA_GATE_DIR", "~/.shishya/gate")).expanduser()


def _read_json(path: Path) -> dict:
    try:
        body = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(body, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return body


def load_config(cwd: str, learner_dir: Path) -> tuple[dict, list[str]]:
    """(merged config, notes). Project rules are added; a project cannot disable rules or lower the bar."""
    user = _read_json(learner_dir / "config.json")
    project = _read_json(project_root(cwd) / ".shishya" / "gate.json")
    notes = []
    for key in set(project) - {"rules"}:
        notes.append(f"project setting {key!r} in .shishya/gate.json is ignored (only 'rules' are allowed there)")
    rules = list(user.get("rules", [])) + list(project.get("rules", []))
    return {"backend": user.get("backend", "prior"),
            "min_probability": float(user.get("min_probability", DEFAULT_MIN_PROBABILITY)),
            "rules": rules, "disable": list(user.get("disable", []))}, notes


def build_schema(config: dict) -> dict:
    schema = copy.deepcopy(SCHEMA)
    disabled = set(config.get("disable", []))
    names = {r["name"] for r in CORE_RULES}
    extra = [r for r in config.get("rules", []) if isinstance(r, dict) and r.get("name") not in names]
    schema["x-shishya"] = {"rules": [r for r in CORE_RULES if r["name"] not in disabled] + extra}
    return schema


def _backend(spec: str) -> Any:
    if spec == "prior":
        return PriorBackend()
    if spec.startswith("http"):
        from ..backends import SystemOneBackend
        return SystemOneBackend(spec, timeout=5, retries=0)
    raise ValueError(f"unknown gate backend {spec!r}: use 'prior' or a decision server URL")


def _objections(rules: Any, answer: dict, ctx: dict) -> list[str]:
    """Messages of the rules that forbid ``answer``. A rule that fails counts as forbidding (fail closed)."""
    out = []
    for r in rules:
        try:
            ok = r(answer, ctx)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            out.append(r.message)
    return list(dict.fromkeys(out))


def gate_decide(tool_name: str, tool_input: dict, cwd: str, *, home: str | None = None) -> dict:
    """Never raises: any failure answers "ask" (fail closed, but never silently blocks work)."""
    learner_dir = default_dir()
    try:
        return _decide(tool_name, tool_input, cwd, home, learner_dir)
    except Exception as e:  # noqa: BLE001 - the Gate must answer
        return {"decision": "ask", "reason": f"could not check this call ({type(e).__name__}: {e}), so it asks",
                "decision_id": None, "learner_dir": str(learner_dir), "forced": True}


def _decide(tool_name: str, tool_input: dict, cwd: str, home: str | None, learner_dir: Path) -> dict:
    from ..engine import Shishya

    config, notes = load_config(cwd, learner_dir)
    text, facts = tool_facts(tool_name, tool_input, cwd, home)
    facts["unparsed"] = not facts["parse_ok"]
    facts["vouchable"] = facts["parse_ok"] and not (facts["inline_code"] or facts["runs_script"])
    backend = _backend(config["backend"])
    if isinstance(backend, PriorBackend):
        backend.read_only = facts["read_only"]
    rule_facts = {k: facts[k] for k in (*FACT_KEYS, "unparsed")}
    schema = build_schema(config)
    engine = Shishya(backend, learner=learner_dir, min_probability=config["min_probability"])
    d = engine.decide(text, schema, facts=rule_facts)
    out = {"decision": None, "reason": "", "decision_id": d.receipt.get("id"), "learner_dir": str(learner_dir),
           "forced": False, "facts": facts, "status": d.status, "confidence": d.confidence,
           "learner": d.receipt.get("learner"), "notes": notes + list(d.reasons) + d.rule_errors}
    # The hard floor does not depend on what the search happened to try: check each rule against each answer.
    ctx = {"input": text, "facts": rule_facts}
    against = {opt: _objections(engine.compile(schema).rules, {"decision": opt}, ctx) for opt in ("allow", "ask")}
    ans = (d.answer or d.suggestion or {}).get("decision")
    if against["ask"] or d.rule_errors:
        out.update(decision="deny", forced=True, reason=" ".join(against["ask"]) or "; ".join(d.rule_errors))
    elif against["allow"]:
        out.update(decision="ask", forced=True, reason=" ".join(against["allow"]))
    elif ans == "allow" and d.status == "ok" and facts["vouchable"]:
        used = (d.receipt.get("learner") or {}).get("used", {}).get("decision")
        why = {"memory": "you approved this exact call before", "head": "it matches calls you approve"}.get(
            used, "the decision model is confident it is routine")
        out.update(decision="allow", reason=f"{why} (confidence {d.confidence:.2f}); no rule objects")
    return out
