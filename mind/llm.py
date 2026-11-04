"""The models the mind talks to. Free models only: anything else is refused."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
STUDENT = ("nvidia/nemotron-3.5-super-vl:free", "dots-studio/dots-3-note-preview:free",
           "nvidia/nemotron-3.5-lightning:free")
TEACHER = ("nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free",
           "nvidia/nemotron-3.5-super-vl:free")


class LLMError(RuntimeError):
    pass


def load_key(path: str | Path | None = None) -> str:
    """The OpenRouter key: from the environment, or the repo's gitignored .env."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise LLMError("no OPENROUTER_API_KEY in the environment or in .env")


class FreeLLM:
    """Tries each free model in turn. A model that is busy or sends an empty reply is skipped."""

    def __init__(self, models: tuple[str, ...] = STUDENT, key: str | None = None, rounds: int = 3,
                 timeout: float = 90, max_tokens: int = 4000):
        for m in models:
            if not m.endswith(":free"):
                raise ValueError(f"only free models are allowed, not {m}")
        self.models, self.key, self.rounds, self.timeout, self.max_tokens = models, key, rounds, timeout, max_tokens
        self.calls = 0

    def __call__(self, prompt: str, system: str = "") -> str:
        key = self.key or load_key()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        last = "no reply"
        for attempt in range(self.rounds):
            for model in self.models:
                try:
                    self.calls += 1
                    r = httpx.post(URL, timeout=self.timeout, headers={"Authorization": f"Bearer {key}"},
                                   json={"model": model, "messages": messages, "temperature": 0.2,
                                         "max_tokens": self.max_tokens})
                    if r.status_code == 200:
                        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
                        if text:
                            return text
                        last = f"{model}: empty reply"
                    else:
                        last = f"{model}: HTTP {r.status_code}"
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                    last = f"{model}: {type(e).__name__}"
            time.sleep(3 * 2 ** attempt)
        raise LLMError(f"the free models did not answer ({last})")


def parse_json(text: str) -> dict[str, Any] | None:
    """The JSON object inside a reply, even with words or code fences around it. None if there is none."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start:end + 1]
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        return value if isinstance(value, dict) else None
    return None
