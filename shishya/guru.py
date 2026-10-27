"""A Guru backed by any chat model on OpenRouter (or another OpenAI-compatible server).

The Guru is asked only when the fast path fails: it gets the text, the fields and the rules the
earlier answers broke, and must return one JSON answer.

    OpenRouterGuru(model="nvidia/nemotron-3.5-super-vl:free")
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"

SYSTEM = (
    "You make decisions. Answer with ONE JSON object whose keys are exactly the field names you are given. "
    "Each value must be exactly one of that field's options (booleans as JSON true or false). "
    "Reply with the JSON object only, no other text. "
    "Lines that start with \"Not allowed\" are hard rules: your answer must respect every one of them."
)


class OpenRouterGuru:
    name = "openrouter"

    def __init__(self, model: str | None = None, *, max_tries: int = 3, api_key: str | None = None,
                 base_url: str = "https://openrouter.ai/api/v1", timeout: float = 60,
                 client: httpx.Client | None = None):
        self.model = model or os.environ.get("SHISHYA_GURU_MODEL") or DEFAULT_MODEL
        self.max_tries = max_tries
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def _prompt(self, text: str, fields: list[dict], feedback: list[str]) -> str:
        parts = [text, "", "Fields to answer:"]
        for f in fields:
            parts.append(f"- {f['name']}: {f.get('instructions') or ''}".rstrip())
            texts = f.get("option_text") or {}
            for o in f["options"]:
                shown = json.dumps(o) if isinstance(o, bool) else str(o)
                parts.append(f"    * {shown}" + (f": {texts[o]}" if texts.get(o) else ""))
        if feedback:
            parts += ["", "Feedback:"] + list(feedback)
        return "\n".join(parts)

    def propose(self, text: str, fields: list[dict], feedback: list[str]) -> dict:
        if not self.api_key:
            raise RuntimeError("the Guru needs an API key: set OPENROUTER_API_KEY or pass api_key")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": self._prompt(text, fields, feedback)}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": "https://github.com/sanattaori/shishya",
                   "X-Title": "Shishya"}
        r = None
        for attempt in range(3):
            try:
                r = self._client.post(self.base_url + "/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                raise RuntimeError(f"Guru request failed: {e}") from e
            if r.status_code != 429 and r.status_code < 500:
                break
            if attempt < 2:
                try:
                    wait = float(r.headers.get("Retry-After", 2 ** attempt))
                except ValueError:
                    wait = 2.0 ** attempt
                time.sleep(min(max(wait, 0.0), 10.0))
        assert r is not None
        if r.status_code != 200:
            raise RuntimeError(f"Guru said HTTP {r.status_code}: {r.text[:200]}")
        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RuntimeError(f"unexpected Guru reply: {r.text[:200]}") from e
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content or "")
        try:
            got = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"the Guru did not return JSON: {content[:200]}") from e
        if not isinstance(got, dict):
            raise TypeError("the Guru did not return a JSON object")
        return got
