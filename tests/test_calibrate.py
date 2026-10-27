"""Calibration: temperature fitting, honest cross-fitted report, CalibratedBackend and the CLI."""
from __future__ import annotations

import json

import pytest

from shishya.backends import FakeBackend
from shishya.calibrate import CalibratedBackend, apply, collect, fit, load, load_labels, report, save
from shishya.cli import main

SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["a", "b"]},
    "urgent": {"type": "boolean", "x-shishya": {"when": {"path": "facts.check_urgent", "eq": True}}},
}}


class Signal:
    """Input "gold=a;k=3": the top option is right except for k in the wrong set; confidence grows with k."""
    name = "signal"
    model = "signal-0"

    def __init__(self, base: float, step: float, wrong_every: int):
        self.base, self.step, self.wrong_every = base, step, wrong_every

    def score(self, text, fields, options):
        gold = text.split("gold=")[1].split(";")[0]
        k = int(text.split("k=")[1].split()[0].split(";")[0])
        pick = ("b" if gold == "a" else "a") if k % self.wrong_every == 0 else gold
        conf = self.base + self.step * (k % 10)
        out = {}
        for f in fields:
            if f.name == "verdict":
                out["verdict"] = {pick: conf, ("b" if pick == "a" else "a"): 1 - conf}
            else:
                out[f.name] = {True: conf, False: 1 - conf} if pick == "a" else {True: 1 - conf, False: conf}
        return out


def examples(n=200, label_urgent=False):
    rows = []
    for k in range(n):
        gold = "a" if k % 3 else "b"
        ans = {"verdict": gold}
        if label_urgent:
            ans["urgent"] = gold == "a"
        rows.append({"input": f"gold={gold};k={k}", "facts": {"check_urgent": True}, "answer": ans})
    return rows


def test_apply_math():
    p = {"x": 0.8, "y": 0.2, "z": 0.0}
    assert apply(p, 1.0) == pytest.approx(p)
    soft = apply(p, 2.0)
    assert soft["x"] == pytest.approx(0.8 ** 0.5 / (0.8 ** 0.5 + 0.2 ** 0.5))
    assert soft["z"] == 0.0 and sum(soft.values()) == pytest.approx(1.0)
    assert apply(p, 0.5)["x"] > 0.8
    with pytest.raises(ValueError):
        apply(p, 0)


def test_overconfident_model_gets_t_above_one_and_lower_ece():
    # right 80% of the time but says 90-99%
    data = collect(Signal(0.90, 0.01, 5), SCHEMA, examples())
    temps = fit(data)
    assert temps["verdict"] > 1.2
    r = report(data, temps)["fields"]["verdict"]
    assert r["ece_after"] < r["ece_before"]
    assert r["n"] == 200 and r["accuracy"] == pytest.approx(0.8)


def test_underconfident_model_gets_t_below_one():
    # right 95% of the time but says 60-78%
    data = collect(Signal(0.60, 0.02, 20), SCHEMA, examples())
    temps = fit(data)
    assert temps["verdict"] < 0.8
    r = report(data, temps)["fields"]["verdict"]
    assert r["ece_after"] < r["ece_before"]


def test_report_coverage_is_monotonic_and_error_is_honest():
    data = collect(Signal(0.55, 0.04, 6), SCHEMA, examples(label_urgent=True))
    rep = report(data, fit(data))
    for table in [rep["all_fields"]["thresholds"]] + [f["thresholds"] for f in rep["fields"].values()]:
        cov = [r["coverage"] for r in table]
        assert cov == sorted(cov, reverse=True)
        assert all(r["error"] is None or 0 <= r["error"] <= 1 for r in table)
    assert [r["threshold"] for r in rep["all_fields"]["thresholds"]] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    # a joint answer is covered only when every field clears the bar, so it never beats one field
    one = {r["threshold"]: r["coverage"] for r in rep["fields"]["verdict"]["thresholds"]}
    for r in rep["all_fields"]["thresholds"]:
        assert r["coverage"] <= one[r["threshold"]] + 1e-12


def test_collect_skips_ineligible_fields_and_bad_labels():
    rows = [{"input": "gold=a;k=1", "facts": {"check_urgent": False}, "answer": {"verdict": "a", "urgent": True}},
            {"input": "gold=a;k=2", "facts": {"check_urgent": True}, "answer": {"verdict": "zzz", "urgent": True}},
            {"input": "gold=a;k=3", "answer": {"urgent": True}}]            # no facts: urgent not asked at all
    data = collect(Signal(0.9, 0.01, 5), SCHEMA, rows)
    assert [d["index"] for d in data] == [0, 1]
    assert set(data[0]["fields"]) == {"verdict"}          # urgent ineligible (when is false)
    assert set(data[1]["fields"]) == {"urgent"}           # verdict label is not an option
    assert data[0]["skipped"] == 1 and data[1]["skipped"] == 1
    with pytest.raises(ValueError, match="no field"):
        collect(FakeBackend(), SCHEMA, [{"input": "x", "answer": {"nope": 1}}])


def test_calibrated_backend_and_files(tmp_path):
    inner = FakeBackend(fixed={"verdict": {"a": 0.9, "b": 0.1}, "urgent": {True: 0.7, False: 0.3}})
    be = CalibratedBackend(inner, {"verdict": 2.0})
    assert be.name == "calibrated" and be.model == "fake-0 (calibrated)"
    from shishya.schema import compile_schema
    comp = compile_schema(SCHEMA)
    out = be.score("t", list(comp.fields), {f.name: f.options for f in comp.fields})
    assert out["verdict"]["a"] == pytest.approx(apply({"a": 0.9, "b": 0.1}, 2.0)["a"])
    assert out["urgent"] == {True: 0.7, False: 0.3}
    path = str(tmp_path / "c.json")
    save(path, {"verdict": 2.0}, {"examples": 3})
    body = json.loads(open(path).read())
    assert body["shishya_calibration"] == 1 and body["meta"]["examples"] == 3 and "created" in body["meta"]
    assert load(path) == {"verdict": 2.0}
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError, match="calibration file"):
        load(str(tmp_path / "bad.json"))


def test_load_labels_errors(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"input": "x", "answer": {"verdict": "a"}}\n\nnot json\n')
    with pytest.raises(ValueError, match="line 3"):
        load_labels(str(p))


def _write(tmp_path, n=40):
    (tmp_path / "s.json").write_text(json.dumps(SCHEMA))
    (tmp_path / "l.jsonl").write_text("\n".join(json.dumps(r) for r in examples(n, True)))


def test_cli_calibrate_end_to_end(tmp_path, capsys):
    _write(tmp_path)
    out = tmp_path / "cal.json"
    code = main(["calibrate", str(tmp_path / "s.json"), str(tmp_path / "l.jsonl"), "--backend", "fake",
                 "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "verdict" in text and "urgent" in text and "All fields" in text and "0.95" in text
    assert set(load(str(out))) == {"verdict", "urgent"}
    assert json.loads(out.read_text())["meta"]["backend"] == "fake"


def test_cli_decide_with_calibration_uses_calibrated_backend(tmp_path, capsys):
    _write(tmp_path)
    cal = tmp_path / "cal.json"
    save(str(cal), {"verdict": 2.0, "urgent": 0.5})
    code = main(["decide", str(tmp_path / "s.json"), "--input", "hello", "--backend", "fake",
                 "--calibration", str(cal), "--facts", '{"check_urgent": true}', "--json"])
    assert code == 0
    d = json.loads(capsys.readouterr().out)
    assert d["receipt"]["backend"] == {"name": "calibrated", "model": "fake-0 (calibrated)"}


def test_cli_serve_wraps_backend(tmp_path, monkeypatch):
    cal = tmp_path / "cal.json"
    save(str(cal), {"verdict": 2.0})
    seen = {}
    monkeypatch.setattr("shishya.server.create_app", lambda be: seen.setdefault("be", be))
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert main(["serve", "--backend", "fake", "--calibration", str(cal)]) == 0
    assert seen["be"].name == "calibrated"
