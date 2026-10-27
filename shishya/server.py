"""HTTP server: ``POST /v1/decide`` and ``GET /healthz`` (needs the ``[server]`` extra).

    POST /v1/decide  {"input": ..., "schema": {...}, "facts": {...}, "rego": "package ...",
                      "min_probability": 0.6}  ->  the Decision as JSON

Python rules are NOT accepted over HTTP, because they are code: only Rego policies are. Python rules
belong to the library, where the caller owns the process.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from . import __version__
from .backends.base import BackendError
from .engine import Shishya
from .rules import RuleError
from .schema import SchemaError
from .when import ConditionError

KEYS = {"input", "schema", "facts", "rego", "min_probability"}


def create_app(engine_factory_or_backend: Callable[[], Shishya] | Any):
    """Build the app from a backend (anything with ``score``) or a function returning a ``Shishya``."""
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    if hasattr(engine_factory_or_backend, "score"):
        engine = Shishya(engine_factory_or_backend)
    else:
        engine = engine_factory_or_backend()

    app = FastAPI(title="Shishya", version=__version__)

    def error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request, exc):
        return error(422, "invalid_request", "the request body must be a JSON object")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "version": __version__}

    @app.post("/v1/decide")
    def decide(payload: dict[str, Any]):
        if not isinstance(payload, dict):
            return error(422, "invalid_request", "the request body must be a JSON object")
        extra = sorted(set(payload) - KEYS)
        if extra:
            return error(422, "invalid_request", f"unknown keys {extra}; allowed: {sorted(KEYS)}")
        if "input" not in payload or "schema" not in payload:
            return error(422, "invalid_request", "'input' and 'schema' are required")
        if not isinstance(payload["schema"], dict):
            return error(422, "invalid_request", "'schema' must be a JSON object")
        rego = payload.get("rego")
        if rego is not None and not isinstance(rego, str):
            return error(422, "invalid_request", "'rego' must be a string")
        mp = payload.get("min_probability")
        if mp is not None and (isinstance(mp, bool) or not isinstance(mp, (int, float)) or not 0 <= mp <= 1):
            return error(422, "invalid_request", "'min_probability' must be a number between 0 and 1")
        eng = engine
        if mp is not None:
            eng = copy.copy(engine)          # shares the schema cache; the shared engine is not changed
            eng.min_probability = float(mp)
        try:
            d = eng.decide(payload["input"], payload["schema"], facts=payload.get("facts"), rego=rego)
        except (SchemaError, ConditionError, RuleError) as e:
            return error(422, "invalid_request", str(e))
        except BackendError as e:
            return error(502, "backend_error", str(e))
        return d.to_dict()

    return app
