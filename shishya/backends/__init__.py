from .base import Backend, BackendError, render

# The rest load on first use, so ``import shishya`` stays fast (httpx alone costs ~60 ms; a hook runs per tool call).
_LAZY = {"ChatLogprobsBackend": ".chat_logprobs", "ClassifierBackend": ".classifier", "FakeBackend": ".fake",
         "RouterBackend": ".classifier", "SystemOneBackend": ".systemone"}

__all__ = ["Backend", "BackendError", "ChatLogprobsBackend", "ClassifierBackend", "FakeBackend", "RouterBackend",
           "SystemOneBackend", "render"]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
