"""Backend for any OpenAI-compatible chat endpoint that returns ``top_logprobs``.

Works with OpenRouter (for example ``liquid/lfm-2.5-2.6b:free``) and local servers such as the
llama.cpp server. One request per field: the options get letters, the model answers with one letter,
and the probability of each option is read from the log-probabilities of the first token.

    ChatLogprobsBackend(base_url="http://127.0.0.1:8080/v1", model="local")
"""
from __future__ import annotations

import math
import os
import string
from typing import Any

import httpx

from ..schema import Field
from .base import BackendError

DEFAULT_MODEL = "liquid/lfm-2.5-2.6b:free"


class ChatLogprobsBackend:
    name = "chat-logprobs"

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", model: str | None = None, *,
                 api_key: str | None = None, timeout: float = 60, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("SHISHYA_SCORER_MODEL") or DEFAULT_MODEL
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def _prompt(self, text: str, f: Field, opts: tuple, letters: list[str]) -> str:
        lines = [text, "", f"Question: {f.instructions or f.name}", ""]
        for letter, o in zip(letters, opts):
            if isinstance(o, bool):
                label = "yes" if o else "no"
            else:
                label = str(o)
                if f.option_text.get(o):
                    label += f": {f.option_text[o]}"
            lines.append(f"{letter}. {label}")
        lines += ["", "Reply with only the letter of the best option."]
        return "\n".join(lines)

    def _ask(self, prompt: str) -> list[dict]:
        body: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        headers.update({"HTTP-Referer": "https://github.com/sanattaori/shishya", "X-Title": "Shishya"})
        try:
            r = self._client.post(self.base_url + "/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise BackendError(f"scorer at {self.base_url} failed: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"scorer said HTTP {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise BackendError(f"the scorer returned no logprobs: {r.text[:300]}") from e

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        out: dict[str, dict[Any, float]] = {}
        for f in fields:
            opts = options[f.name]
            if len(opts) > 26:
                raise BackendError(f"field {f.name!r} has {len(opts)} options; at most 26 are supported")
            letters = list(string.ascii_uppercase[:len(opts)])
            top = self._ask(self._prompt(text, f, opts, letters))
            mass = dict.fromkeys(letters, 0.0)
            for t in top:
                try:
                    tok = str(t["token"]).strip().upper()
                    if tok in mass:
                        mass[tok] += math.exp(float(t["logprob"]))
                except (KeyError, TypeError, ValueError) as e:
                    raise BackendError(f"unexpected logprob entry: {str(t)[:200]}") from e
            total = sum(mass.values())
            if total > 0:
                out[f.name] = {o: mass[letter] / total for letter, o in zip(letters, opts)}
            else:
                out[f.name] = {o: 1.0 / len(opts) for o in opts}
        return out
