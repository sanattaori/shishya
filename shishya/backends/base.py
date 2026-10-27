"""What a model backend must do: give a probability for every option of every field."""
from __future__ import annotations

import json
from typing import Any, Protocol

from ..schema import Field

FACTS_HEADER = "\n\nKnown facts:\n"     # separates the input from the facts in the text a model reads


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    name: str
    model: str

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        """Probabilities per field over ``options[field.name]`` (the options still allowed).

        ``text`` is what the model reads (the input, plus the facts when they are shown).
        """
        ...


def render(input: Any, facts: Any = None, show_facts: bool = True) -> str:
    """The text the model reads: the input verbatim (JSON for structures), then the facts."""
    text = input if isinstance(input, str) else json.dumps(input, ensure_ascii=False, indent=1)
    if show_facts and facts not in (None, {}, []):
        known = facts if isinstance(facts, str) else json.dumps(facts, ensure_ascii=False, indent=1)
        text = f"{text}{FACTS_HEADER}{known}"
    return text
