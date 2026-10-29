"""Shishya Gate: hard rules for a coding agent's tool calls, and fewer prompts as it learns what you approve."""
from __future__ import annotations

__all__ = ["gate_decide"]


def __getattr__(name: str):
    if name == "gate_decide":     # imported lazily, so the hook module loads even while the policy is unavailable
        from .policy import gate_decide
        return gate_decide
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
