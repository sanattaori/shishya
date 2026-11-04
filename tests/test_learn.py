import json
import random
import time

import pytest

from shishya import Learner, Shishya, rule
from shishya.backends import FakeBackend
from shishya.learn import features, text_key, train_head

SCHEMA = {"type": "object", "properties": {
    "verdict": {"enum": ["approve", "deny"]},
    "urgent": {"type": "boolean"},
}}
VERDICT = {"approve": 0.7, "deny": 0.3}


def engine(tmp_path, fixed=None, **kw):
    backend = FakeBackend(fixed=fixed or {"verdict": VERDICT, "urgent": {True: 0.2, False: 0.8}})
    return Shishya(backend, learner=Learner(tmp_path / "l", **kw.pop("learner_kw", {})), **kw)


def test_no_learner_unchanged_except_id():
    s = Shishya(FakeBackend(fixed={"verdict": VERDICT, "urgent": {True: 0.2, False: 0.8}}))
    d = s.decide("hello", SCHEMA)
    assert len(d.receipt["id"]) == 32
    assert "learner" not in d.receipt
    assert d.answer == {"verdict": "approve", "urgent": False}
    json.dumps(d.to_dict())
    with pytest.raises(ValueError):
        s.correct(d, {"verdict": "deny"})


def test_receipt_with_learner(tmp_path):
    s = engine(tmp_path)
    d = s.decide("hello", SCHEMA)
    assert d.receipt["learner"] == {"version": 0, "used": {"verdict": None, "urgent": None}}
    json.dumps(d.to_dict())


def test_memory_from_one_correction(tmp_path):
    s = engine(tmp_path)
    d = s.decide("Need 5 days off", SCHEMA)
    rep = s.correct(d, {"verdict": "deny"})
    assert rep["recorded"] == ["verdict"] and rep["skipped"] == {}
    d2 = s.decide("need  5 DAYS off ", SCHEMA)
    assert d2.fields["verdict"]["options"]["deny"] >= 0.8
    assert d2.receipt["learner"]["used"] == {"verdict": "memory", "urgent": None}
    assert d2.answer["verdict"] == "deny"
    s.correct(d2, {"verdict": "deny"})
    d3 = s.decide("Need 5 days off", SCHEMA)
    assert d3.fields["verdict"]["options"]["deny"] >= 0.889
    # a newer conflicting correction flips it
    s.correct(d3, {"verdict": "approve"})
    s.correct(s.decide("Need 5 days off", SCHEMA), {"verdict": "approve"})
    s.correct(s.decide("Need 5 days off", SCHEMA), {"verdict": "approve"})
    d4 = s.decide("Need 5 days off", SCHEMA)
    assert d4.answer["verdict"] == "approve"
    assert d4.fields["verdict"]["options"]["approve"] >= 0.889


def test_other_text_does_not_use_memory(tmp_path):
    s = engine(tmp_path)
    s.correct(s.decide("Need 5 days off", SCHEMA), {"verdict": "deny"})
    d = s.decide("Need 6 days off", SCHEMA)
    assert d.receipt["learner"]["used"]["verdict"] is None
    assert d.fields["verdict"]["options"]["deny"] == pytest.approx(0.3)


def test_rules_still_win(tmp_path):
    @rule(uses=["verdict"])
    def never_deny(ans, ctx):
        "No denials."
        return ans["verdict"] != "deny"

    s = engine(tmp_path, rules=[never_deny])
    s.correct(s.decide("Need 5 days off", SCHEMA), {"verdict": "deny"})
    d = s.decide("Need 5 days off", SCHEMA)
    assert d.top["verdict"] == "deny"
    assert d.answer["verdict"] == "approve"
    assert d.status == "ok"


def test_skips(tmp_path):
    schema = {"type": "object", "properties": {
        "verdict": {"enum": ["approve", "deny"]},
        "urgent": {"type": "boolean"},
        "note": {"enum": ["a"]},
    }}
    s = engine(tmp_path)
    d = s.decide("x", schema)
    rep = s.correct(d, {"note": "a", "verdict": "maybe", "urgent": 1, "nope": "x"})
    assert rep["recorded"] == [] and rep["train"] is None
    assert set(rep["skipped"]) == {"note", "verdict", "urgent", "nope"}
    assert s.correct(d, {"urgent": True})["recorded"] == ["urgent"]
    with pytest.raises(ValueError):
        s.learner.correct("nope", {"verdict": "deny"})


def test_cross_process(tmp_path):
    s = engine(tmp_path)
    d = s.decide("Need 5 days off", SCHEMA)
    other = Learner(tmp_path / "l")
    assert other.correct(d.receipt["id"], {"verdict": "deny"})["recorded"] == ["verdict"]
    d2 = s.decide("Need 5 days off", SCHEMA)
    assert d2.receipt["learner"]["used"]["verdict"] == "memory"
    assert oct((tmp_path / "l").stat().st_mode & 0o777) == "0o700"
    assert oct((tmp_path / "l" / "learn.db").stat().st_mode & 0o777) == "0o600"


def _teach(s, texts, label, source="human"):
    for t in texts:
        s.correct(s.decide(t, SCHEMA), {"verdict": label(t)}, source=source, train=False)


def _texts(n, rng, contractor_rate=0.5):
    words = [f"w{i}" for i in range(60)]
    out = []
    for i in range(n):
        ws = rng.sample(words, 6)
        if rng.random() < contractor_rate:
            ws.append("contractor")
        rng.shuffle(ws)
        out.append(f"request {i} " + " ".join(ws))
    return out


def test_head_promoted_and_applied(tmp_path):
    s = engine(tmp_path)
    rng = random.Random(1)
    _teach(s, _texts(300, rng), lambda t: "deny" if "contractor" in t else "approve")
    rep = s.learner.train()
    r = rep["fields"]["verdict"]
    assert r["promoted"], r["why"]
    assert r["n_test"] >= 5 and r["nll_gain_ci"][0] > 0
    assert rep["version"] == 1
    d = s.decide("request 9999 w1 w2 contractor w3", SCHEMA)
    assert d.receipt["learner"]["used"]["verdict"] == "head"
    assert d.receipt["learner"]["version"] == 1
    assert d.answer["verdict"] == "deny"
    d = s.decide("request 9998 w1 w2 w4 w3", SCHEMA)
    assert d.answer["verdict"] == "approve"
    assert "learner v1" in s.learner.wrap(s.backend).model
    st = s.learner.status()
    assert st["fields"]["verdict"]["promoted"] and st["version"] == 1 and st["last_report"]["version"] == 1


def test_head_not_promoted_on_noise(tmp_path):
    s = Shishya(FakeBackend(fixed={"verdict": {"approve": 0.5, "deny": 0.5}, "urgent": {True: 0.5, False: 0.5}}),
                learner=Learner(tmp_path / "l"))
    rng = random.Random(2)
    _teach(s, _texts(200, rng), lambda t: rng.choice(["approve", "deny"]))
    r = s.learner.train()["fields"]["verdict"]
    assert not r["promoted"] and "base model is used" in r["why"]
    assert s.learner.version == 0
    d = s.decide("request 9999 w1 contractor", SCHEMA)
    assert d.receipt["learner"]["used"]["verdict"] is None


def test_few_labels_no_head(tmp_path):
    s = engine(tmp_path)
    rep = s.correct(s.decide("a", SCHEMA), {"verdict": "deny"})
    assert "more labels" in rep["train"]["fields"]["verdict"]["why"]


def test_test_split_not_used_for_training(tmp_path, monkeypatch):
    import shishya.learn as learn
    seen = []
    real = learn.train_head
    monkeypatch.setattr(learn, "train_head", lambda rows: (seen.extend(rows), real(rows))[1])
    s = engine(tmp_path)
    rng = random.Random(3)
    texts = _texts(80, rng)
    _teach(s, texts, lambda t: "approve")
    # approvals from "model" or other sources never go to the test split
    _teach(s, ["extra one", "extra two", "extra three", "extra four", "extra five", "extra six"], lambda t: "deny",
           source="auto")
    rep = s.learner.train()["fields"]["verdict"]
    test_keys = {k for k in map(text_key, texts + ["extra one"]) if int(k[:8], 16) % 5 == 0}
    assert rep["n_test"] == sum(int(text_key(t)[:8], 16) % 5 == 0 for t in texts)
    assert rep["n_train"] == len(texts) + 6 - rep["n_test"]
    assert len(seen) == rep["n_train"]
    assert test_keys  # the split is not empty


def test_rollback(tmp_path):
    s = engine(tmp_path)
    rng = random.Random(1)
    texts = _texts(300, rng)
    _teach(s, texts, lambda t: "deny" if "contractor" in t else "approve")
    assert s.learner.train()["version"] == 1
    # more labels change the weights, so a new version is written
    _teach(s, _texts(60, random.Random(5)), lambda t: "deny" if "contractor" in t else "approve")
    assert s.learner.train()["version"] == 2
    v1 = json.loads((tmp_path / "l" / "versions" / "model-1.json").read_text())
    # a rollback is saved as a new version, so a version number in a receipt always means one model
    assert s.learner.rollback() == 3
    assert s.learner.version == 3
    assert s.learner.model()["heads"] == v1["heads"] and s.learner.model()["rolled_back_to"] == 1
    with pytest.raises(ValueError):
        s.learner.rollback()            # nothing before version 1


def test_keep_text_false(tmp_path):
    s = engine(tmp_path, learner_kw={"keep_text": False})
    rng = random.Random(1)
    _teach(s, _texts(300, rng), lambda t: "deny" if "contractor" in t else "approve")
    assert s.learner.train()["fields"]["verdict"]["promoted"]
    import sqlite3
    db = sqlite3.connect(tmp_path / "l" / "learn.db")
    assert db.execute("SELECT COUNT(*) FROM decisions WHERE text IS NOT NULL").fetchone()[0] == 0
    db.close()
    s.correct(s.decide("again", SCHEMA), {"verdict": "deny"})
    assert s.decide("again", SCHEMA).receipt["learner"]["used"]["verdict"] == "memory"


def test_features_are_stable():
    assert features("a b") == features("A  B")
    assert len(features("x " * 1000)) <= 3


def test_train_speed():
    rng = random.Random(0)
    rows = []
    for _ in range(1000):
        base = [rng.random() for _ in range(3)]
        z = sum(base)
        rows.append({"options": ['"a"', '"b"', '"c"'], "base": [x / z for x in base],
                     "feats": [rng.randrange(2**20) for _ in range(20)], "y": rng.randrange(3)})
    t = time.perf_counter()
    train_head(rows)
    ms = (time.perf_counter() - t) * 1000
    print(f"train_ms {ms:.0f}")
    assert ms < 3000


def test_broken_model_falls_back_to_base(tmp_path):
    """A corrupt model file must not break decisions (a crash would stop the caller)."""
    s = engine(tmp_path)
    s.decide("hello", SCHEMA)
    (tmp_path / "l" / "model.json").write_text("{not json")
    d = s.decide("hello", SCHEMA)
    assert d.answer == {"verdict": "approve", "urgent": False}
    assert any(r.startswith("learner error (base model used)") for r in d.reasons)


def test_one_answer_only_is_never_promoted(tmp_path):
    """Only one answer seen (say, only approvals): the test cannot show the head tells cases apart, so no head."""
    s = engine(tmp_path)
    _teach(s, _texts(300, random.Random(1)), lambda t: "approve")
    rep = s.learner.train()["fields"]["verdict"]
    assert rep["n_test"] >= 5 and not rep["promoted"]
    assert "same answer" in rep["why"]
