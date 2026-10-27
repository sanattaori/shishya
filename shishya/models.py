"""Shishya's trained specialist models: download once, verify, cache.

    from shishya import models
    models.path("news-base")      # -> local folder with the model (downloads ~280 MB the first time)

Or in a backend: ``ClassifierBackend("shishya:news-base", fields=...)``. Files are verified with SHA-256
before use and kept in ``$SHISHYA_HOME/models`` (default ``~/.cache/shishya/models``).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

RELEASES = "https://github.com/sanattaori/shishya/releases/download"

MODELS = {
    "news-base": {
        "file": "shishya-news-base-v1.zip", "release": "v0.2.0", "mb": 276,
        "sha256": "974a7ed18f8babc1ec6222ebb8f8cec47fee0782e9ad6b51a11497c94eca7325",
        "about": "Financial news headline -> bearish/bullish/neutral. ModernBERT-base, 17 ms on a laptop. "
                 "Indian news test macro-F1 0.889.",
        "labels": ["bearish", "bullish", "neutral"],
    },
    "news-large": {
        "file": "shishya-news-large-v1.zip", "release": "v0.2.0", "mb": 730,
        "sha256": "714827e06bca7ba0c02cb1056b985601e40e83c046a064c5ce235a1c11df657a",
        "about": "Same task, ModernBERT-large: a little more accurate (Indian news test macro-F1 0.895), "
                 "2.6x bigger, 2x slower.",
        "labels": ["bearish", "bullish", "neutral"],
    },
}


class ModelError(RuntimeError):
    pass


def home() -> Path:
    return Path(os.environ.get("SHISHYA_HOME", Path.home() / ".cache" / "shishya")) / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ModelError(f"refusing to extract {member.filename!r} outside the model folder")
    zf.extractall(dest)


def path(name: str, *, download: bool = True) -> str:
    """Local folder of a registered model, downloading and verifying it if needed."""
    if name not in MODELS:
        raise ModelError(f"unknown model {name!r}; available: {', '.join(sorted(MODELS))}")
    info = MODELS[name]
    dest = home() / info["file"].removesuffix(".zip")
    if (dest / "config.json").exists():
        return str(dest)
    if not download:
        raise ModelError(f"model {name!r} is not downloaded yet")
    import httpx

    url = f"{RELEASES}/{info['release']}/{info['file']}"
    home().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home()) as tmp:
        zpath = Path(tmp) / info["file"]
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            if r.status_code != 200:
                raise ModelError(f"download of {url} failed: HTTP {r.status_code}")
            with zpath.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        got = _sha256(zpath)
        if got != info["sha256"]:
            raise ModelError(f"checksum mismatch for {info['file']}: got {got}, expected {info['sha256']}")
        unpacked = Path(tmp) / "unpacked"
        with zipfile.ZipFile(zpath) as zf:
            _safe_extract(zf, unpacked)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(unpacked), dest)
    return str(dest)


def resolve_model(ref: str) -> str:
    """``shishya:<name>`` -> local folder of that registered model; anything else is returned unchanged."""
    return path(ref.split(":", 1)[1]) if ref.startswith("shishya:") else ref
