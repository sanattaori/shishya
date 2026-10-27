"""A deterministic stand-in for a model: no weights, no network. For tests, CI and trying the API.

Probabilities come from a hash of (text, field, option), so they are stable but meaningless, unless
``fixed`` gives them explicitly: ``FakeBackend(fixed={"action": {"hold": 0.7, "watch": 0.3}})``.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..schema import Field
from .base import BackendError


class FakeBackend:
    name = "fake"

    def __init__(self, fixed: dict[str, dict[Any, float]] | None = None, model: str = "fake-0"):
        self.fixed = fixed or {}
        self.model = model
        self.calls = 0

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        self.calls += 1
        out = {}
        for f in fields:
            opts = options[f.name]
            if f.name in self.fixed:
                given = self.fixed[f.name]
                missing = [o for o in opts if o not in given]
                if missing:
                    raise BackendError(f"fixed probabilities for {f.name!r} miss options {missing}")
                out[f.name] = {o: float(given[o]) for o in opts}
                continue
            w = {}
            for o in opts:
                h = hashlib.sha256(f"{text}|{f.name}|{o}".encode()).digest()
                w[o] = 0.05 + int.from_bytes(h[:4], "big") / 2**32
            total = sum(w.values())
            out[f.name] = {o: v / total for o, v in w.items()}
        return out
