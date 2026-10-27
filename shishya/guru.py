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


def build_prompt(text: str, fields: list[dict], feedback: list[str]) -> str:
    """The user message for a Guru: the text, the fields with their options, then the feedback lines."""
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


class OpenRouterGuru:
    name = "openrouter"
    trainable = False       # set True only for models whose license allows training on their outputs

    def __init__(self, model: str | None = None, *, max_tries: int = 3, api_key: str | None = None,
                 base_url: str = "https://openrouter.ai/api/v1", timeout: float = 60,
                 client: httpx.Client | None = None):
        self.model = model or os.environ.get("SHISHYA_GURU_MODEL") or DEFAULT_MODEL
        self.max_tries = max_tries
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def propose(self, text: str, fields: list[dict], feedback: list[str]) -> dict:
        if not self.api_key:
            raise RuntimeError("the Guru needs an API key: set OPENROUTER_API_KEY or pass api_key")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": build_prompt(text, fields, feedback)}],
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


class ClaudeCodeGuru:
    """A Guru that asks Claude through the local Claude Code CLI (``claude -p``), with your own login.

    Uses whatever account the ``claude`` command is logged in to, so no API key is needed. Claude runs
    with no tools, no MCP servers and no saved session, and its answer must match a JSON Schema built
    from the fields. Answers from Claude are marked not trainable: Shishya never learns from them.

        ClaudeCodeGuru(model="sonnet")
    """

    name = "claude-code"
    trainable = False

    def __init__(self, model: str | None = None, *, max_tries: int = 2, command: str = "claude",
                 timeout: float = 120):
        self.model = model or os.environ.get("SHISHYA_CLAUDE_MODEL") or "sonnet"
        self.max_tries = max_tries
        self.command = command
        self.timeout = timeout

    @staticmethod
    def _schema(fields: list[dict]) -> dict:
        props = {}
        for f in fields:
            if f["kind"] == "boolean":
                props[f["name"]] = {"type": "boolean"}
            else:
                props[f["name"]] = {"type": "string", "enum": list(f["options"])}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def propose(self, text: str, fields: list[dict], feedback: list[str]) -> dict:
        import shutil
        import subprocess
        import tempfile

        exe = shutil.which(self.command)
        if not exe:
            raise RuntimeError(f"the Claude Code CLI ({self.command!r}) is not on PATH")
        cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--model", self.model, "--system-prompt", SYSTEM,
               "--json-schema", json.dumps(self._schema(fields))]
        prompt = build_prompt(text, fields, feedback)
        with tempfile.TemporaryDirectory(prefix="shishya-guru-") as cwd:  # no project files in view
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd,
                                  timeout=self.timeout, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {(proc.stderr or proc.stdout).strip()[:300]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p did not return JSON: {proc.stdout[:200]}") from e
        if data.get("is_error"):
            raise RuntimeError(f"claude -p error: {str(data.get('result'))[:300]}")
        got = data.get("structured_output")
        if got is None:
            got = json.loads(re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", data.get("result") or ""))
        if not isinstance(got, dict):
            raise TypeError("Claude did not return a JSON object")
        return got
