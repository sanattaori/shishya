"""Backend for a local text classifier (a Hugging Face sequence-classification model) on chosen fields.

A small classifier trained for one judgement (for example: is this news bearish, bullish or neutral?)
can be far better at it than a general decision model. Map each option of the field to one of the
classifier's labels:

    ClassifierBackend("shishya:news-base", fields={"news_impact": {"bad": "bearish", "good": "bullish",
                                                                   "routine": "neutral"}})

``shishya:<name>`` downloads one of Shishya's trained models (see ``shishya.models``); a local folder or
a Hugging Face model id works too.

Usually combined with a general model through ``RouterBackend``. The classifier reads only the input,
not the facts block (it was trained on bare text). If the model's config has ``shishya_temperature``
(written by Shishya's training scripts), it is applied so the probabilities are calibrated.

Needs ``pip install 'shishya[local]'`` (torch and transformers).
"""
from __future__ import annotations

from typing import Any

from ..schema import Field
from .base import FACTS_HEADER, BackendError


class ClassifierBackend:
    name = "classifier"

    def __init__(self, model: str, *, fields: dict[str, dict[Any, str]], device: str | None = None,
                 temperature: float | None = None, max_length: int = 128):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise BackendError("ClassifierBackend needs torch and transformers: pip install 'shishya[local]'") from e
        if not fields:
            raise BackendError("ClassifierBackend needs at least one field mapping")
        from ..models import ModelError, resolve_model
        try:
            model = resolve_model(model)            # "shishya:news-base" -> downloaded, verified folder
        except ModelError as e:
            raise BackendError(str(e)) from e
        self._torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.net = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.net.float()
        labels = {int(k): v for k, v in self.net.config.id2label.items()}
        self.label_index = {v: k for k, v in labels.items()}
        for fname, mapping in fields.items():
            if len(set(mapping.values())) != len(mapping):
                raise BackendError(f"field {fname!r}: each option needs its own classifier label")
            bad = [lab for lab in mapping.values() if lab not in self.label_index]
            if bad:
                raise BackendError(f"field {fname!r}: labels {bad} are not in the model ({sorted(self.label_index)})")
        self.fields = fields
        self.temperature = float(temperature if temperature is not None
                                 else getattr(self.net.config, "shishya_temperature", 1.0))
        self.max_length = max_length
        self.model = str(model)

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        bare = text.split(FACTS_HEADER, 1)[0].strip()
        torch = self._torch
        with torch.no_grad():
            enc = self.tok(bare, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.net(**enc).logits[0].float() / self.temperature, -1).cpu().tolist()
        out = {}
        for f in fields:
            mapping = self.fields.get(f.name)
            if mapping is None:
                raise BackendError(f"ClassifierBackend has no mapping for field {f.name!r}")
            missing = [o for o in options[f.name] if o not in mapping]
            if missing:
                raise BackendError(f"field {f.name!r}: options {missing} have no classifier label")
            out[f.name] = {o: probs[self.label_index[mapping[o]]] for o in options[f.name]}
        return out


class RouterBackend:
    """Send each field to the backend that is best at it; one call per backend used.

        RouterBackend(default=SystemOneBackend(url), routes={"news_impact": news_classifier})
    """

    name = "router"

    def __init__(self, default: Any, routes: dict[str, Any]):
        self.default = default
        self.routes = dict(routes)

    @property
    def model(self) -> str:
        parts = [f"default={getattr(self.default, 'model', self.default.name)}"]
        parts += [f"{k}={getattr(b, 'model', b.name)}" for k, b in self.routes.items()]
        return ", ".join(parts)

    def score(self, text: str, fields: list[Field], options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        groups: dict[int, tuple[Any, list[Field]]] = {}
        for f in fields:
            b = self.routes.get(f.name, self.default)
            groups.setdefault(id(b), (b, []))[1].append(f)
        out: dict[str, dict[Any, float]] = {}
        for b, fs in groups.values():
            out.update(b.score(text, fs, {f.name: options[f.name] for f in fs}))
        return out
