"""Model registry: verified downloads, safe unpacking, local cache."""
import hashlib
import io
import zipfile

import pytest

from shishya import models


def make_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("SHISHYA_HOME", str(tmp_path))
    good = make_zip({"config.json": "{}", "README.md": "card"})
    reg = {"tiny": {"file": "tiny-v1.zip", "release": "v0", "mb": 0, "about": "", "labels": [],
                    "sha256": hashlib.sha256(good).hexdigest()}}
    monkeypatch.setattr(models, "MODELS", reg)
    return good, reg


def serve(monkeypatch, payload, status=200):
    class Resp:
        status_code = status

        def iter_bytes(self, n):
            yield payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    import httpx
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Resp())


def test_download_verifies_and_caches(fake_registry, monkeypatch, tmp_path):
    good, _ = fake_registry
    serve(monkeypatch, good)
    p = models.path("tiny")
    assert (tmp_path / "models" / "tiny-v1" / "config.json").exists() and p.endswith("tiny-v1")
    serve(monkeypatch, b"never fetched again", status=500)
    assert models.path("tiny") == p                         # cached
    assert models.resolve_model("shishya:tiny") == p
    assert models.resolve_model("some/hf-id") == "some/hf-id"


def test_checksum_mismatch_is_refused(fake_registry, monkeypatch):
    serve(monkeypatch, make_zip({"config.json": "{} tampered"}))
    with pytest.raises(models.ModelError, match="checksum"):
        models.path("tiny")


def test_zip_slip_is_refused(fake_registry, monkeypatch):
    evil = make_zip({"../escape.txt": "x", "config.json": "{}"})
    models.MODELS["tiny"]["sha256"] = hashlib.sha256(evil).hexdigest()
    serve(monkeypatch, evil)
    with pytest.raises(models.ModelError, match="outside"):
        models.path("tiny")


def test_unknown_model_and_offline():
    with pytest.raises(models.ModelError, match="unknown model"):
        models.path("nope")


def test_registry_entries_are_complete():
    for name, info in models.MODELS.items():
        assert len(info["sha256"]) == 64 and info["file"].endswith(".zip") and info["labels"], name
