"""Compile a JSON Schema subset into an ordered list of decision fields.

Supported in v0.1: a top-level ``object`` whose properties are

* ``{"type": "boolean"}``                              -> a yes/no decision
* ``{"type": "string", "enum": ["a", "b", ...]}``      -> a choice between the listed values

Each property may carry ``description`` (what the model is asked) and an ``x-shishya`` block:

    "x-shishya": {
      "options": {"approve": "Grant the leave as asked", ...},  # text shown to the model per option
      "when":    <condition>,                                   # field is decided only if this holds
      "option_when": {"approve": <condition>, ...}              # option is allowed only if this holds
    }

Conditions are evaluated by plain code against input/facts (see ``when.py``). A field whose
``when`` does not hold is not asked and comes back as ``null``. Fields are decided in the order of
``x-shishya.order`` at the top level, or in property order. Top-level ``x-shishya.rules`` holds
declarative rules that read ``answer.*`` and ``facts.*`` (see ``rules.json_rule``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from . import when as when_mod

EXT = "x-shishya"


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Field:
    name: str
    kind: str                          # "choice" | "noul"
    options: tuple                     # choice: str values; noul: (True, False)
    instructions: str
    option_text: dict = field(default_factory=dict)    # option -> description (choice only)
    when: Any = None
    option_when: dict = field(default_factory=dict)    # option -> condition


@dataclass(frozen=True)
class Compiled:
    fields: tuple[Field, ...]
    digest: str
    schema: dict
    rules: tuple = ()               # declarative rules from x-shishya.rules (shishya.rules.Rule)

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.fields]


def digest(obj: Any) -> str:
    """Content hash. Key order is kept on purpose: property order decides search and answer order."""
    return "sha256:" + hashlib.sha256(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().capitalize() + "?"


def _no_answer(cond: Any, where: str) -> None:
    # field conditions run before the model is asked, so there is no answer to read yet
    if any(p.split(".")[0] == "answer" for p in when_mod.paths(cond)):
        raise SchemaError(f"{where}: field conditions cannot read 'answer' (put that in x-shishya.rules)")


def _field(name: str, prop: Any) -> Field:
    where = f"properties.{name}"
    if not isinstance(prop, dict):
        raise SchemaError(f"{where} must be an object")
    ext = prop.get(EXT, {})
    if not isinstance(ext, dict):
        raise SchemaError(f"{where}.{EXT} must be an object")
    unknown = set(ext) - {"options", "when", "option_when"}
    if unknown:
        raise SchemaError(f"{where}.{EXT} has unknown keys {sorted(unknown)}")
    desc = prop.get("description")
    if desc is not None and not isinstance(desc, str):
        raise SchemaError(f"{where}.description must be a string")
    instructions = desc.strip() if desc and desc.strip() else _humanize(name)

    cond = ext.get("when")
    if cond is not None:
        try:
            when_mod.check(cond)
        except when_mod.ConditionError as e:
            raise SchemaError(f"{where}.{EXT}.when: {e}") from e
        _no_answer(cond, f"{where}.{EXT}.when")

    typ = prop.get("type")
    if typ == "boolean":
        if "enum" in prop or "options" in ext or "option_when" in ext:
            raise SchemaError(f"{where}: a boolean takes no enum/options/option_when")
        return Field(name, "noul", (True, False), instructions, {}, cond, {})

    enum = prop.get("enum")
    if typ not in (None, "string") or enum is None:
        raise SchemaError(f"{where}: v0.1 supports only boolean and string enum properties")
    if not isinstance(enum, list) or len(enum) < 1:
        raise SchemaError(f"{where}.enum must be a non-empty list")
    if not all(isinstance(v, str) and v for v in enum):
        raise SchemaError(f"{where}.enum values must be non-empty strings")
    if len(set(enum)) != len(enum):
        raise SchemaError(f"{where}.enum has duplicate values")
    if len(enum) > 255:
        raise SchemaError(f"{where}.enum has more than 255 values")

    opt_text = ext.get("options", {})
    if not isinstance(opt_text, dict) or not all(isinstance(v, str) for v in opt_text.values()):
        raise SchemaError(f"{where}.{EXT}.options must map option -> text")
    extra = set(opt_text) - set(enum)
    if extra:
        raise SchemaError(f"{where}.{EXT}.options names values not in enum: {sorted(extra)}")

    opt_when = ext.get("option_when", {})
    if not isinstance(opt_when, dict):
        raise SchemaError(f"{where}.{EXT}.option_when must be an object")
    extra = set(opt_when) - set(enum)
    if extra:
        raise SchemaError(f"{where}.{EXT}.option_when names values not in enum: {sorted(extra)}")
    for opt, c in opt_when.items():
        try:
            when_mod.check(c)
        except when_mod.ConditionError as e:
            raise SchemaError(f"{where}.{EXT}.option_when.{opt}: {e}") from e
        _no_answer(c, f"{where}.{EXT}.option_when.{opt}")

    return Field(name, "choice", tuple(enum), instructions, dict(opt_text), cond, dict(opt_when))


def compile_schema(schema: Any) -> Compiled:
    if not isinstance(schema, dict):
        raise SchemaError("schema must be a JSON object")
    if schema.get("type") != "object":
        raise SchemaError('schema must have "type": "object" at the top level')
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        raise SchemaError("schema needs a non-empty 'properties' object")
    fields = {name: _field(name, prop) for name, prop in props.items()}

    top = schema.get(EXT, {})
    if not isinstance(top, dict):
        raise SchemaError(f"top-level {EXT} must be an object")
    unknown = set(top) - {"order", "rules"}
    if unknown:
        raise SchemaError(f"top-level {EXT} has unknown keys {sorted(unknown)}")
    order = top.get("order")
    if order is None:
        names = list(props)
    else:
        if not isinstance(order, list) or sorted(order) != sorted(props) or len(set(order)) != len(order):
            raise SchemaError(f"{EXT}.order must list every property exactly once")
        names = order
    from .rules import RuleError, json_rule
    specs = top.get("rules", [])
    if not isinstance(specs, list):
        raise SchemaError(f"{EXT}.rules must be a list")
    try:
        rules = tuple(json_rule(r, set(props)) for r in specs)
    except RuleError as e:
        raise SchemaError(f"{EXT}.rules: {e}") from e
    if len({r.name for r in rules}) != len(rules):
        raise SchemaError(f"{EXT}.rules: rule names must be unique")
    return Compiled(tuple(fields[n] for n in names), digest(schema), schema, rules)
