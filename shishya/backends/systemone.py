"""Backend for any server that speaks the System One decision API (``POST /v1/systemone``).

Works with open decision servers such as Kev, jev-style and other Jev-compatible servers. All
fields go in one request, so the model reads the text once.

    SystemOneBackend("http://127.0.0.1:8009")
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..schema import Field
from .base import BackendError


class SystemOneBackend:
    name = "systemone"

    def __init__(self, url: str = "http://127.0.0.1:8009", *, model: str | None = None,
                 api_key: str | None = None, timeout: float = 120, retries: int = 2):
        self.url = url.rstrip("/")
        self.model = model or "default"
        self.api_key = api_key if api_key is not None else os.environ.get("SHISHYA_BACKEND_KEY")
        self.timeout = timeout
        self.retries = retries
        self._client = httpx.Client(timeout=timeout)

    def _question(self, f: Field, opts: tuple) -> dict:
        if f.kind == "noul":
            return {"type": "noul", "instructions": f.instructions}
        return {"type": "choice", "instructions": f.instructions,
                "criteria": {o: f.option_text.get(o) for o in opts}}

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        body: dict[str, Any] = {"state": text,
                                "questions": {f.name: self._question(f, options[f.name]) for f in fields}}
        if self.model != "default":
            body["model"] = self.model
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(self.url + "/v1/systemone", json=body, headers=headers)
            except httpx.HTTPError as e:
                last = e
            else:
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError as e:
                        raise BackendError(f"decision server returned non-JSON: {r.text[:200]}") from e
                    return self._parse(data, fields, options)
                if r.status_code < 500 and r.status_code != 429:
                    raise BackendError(f"decision server said HTTP {r.status_code}: {r.text[:300]}")
                last = BackendError(f"HTTP {r.status_code}: {r.text[:200]}")
            if attempt < self.retries:
                time.sleep(0.5 * 2 ** attempt)
        raise BackendError(f"decision server at {self.url} failed: {last}")

    def _parse(self, data: dict, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        try:
            answers = data["answers"]
            if isinstance(data.get("model"), str) and self.model == "default":
                self.model = data["model"]
            out = {}
            for f in fields:
                a = answers[f.name]
                if f.kind == "noul":
                    if "probabilities" in a:
                        p_true = float(a["probabilities"]["true"])
                    else:
                        p_true = float(a["noul"])
                    out[f.name] = {True: p_true, False: 1.0 - p_true}
                else:
                    probs = a["probabilities"]
                    out[f.name] = {o: float(probs[o]) for o in options[f.name]}
            return out
        except (KeyError, TypeError, ValueError) as e:
            raise BackendError(f"unexpected answer from decision server: {str(data)[:300]}") from e
