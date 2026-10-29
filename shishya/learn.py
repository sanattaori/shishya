"""Learn from corrections: instant memory plus a small head that is used only when it is proven better.

    s = Shishya(backend, rules=[...], learner="~/.shishya/leave")
    d = s.decide(text, schema, facts=facts)
    s.correct(d, {"verdict": "deny"})       # records the label and retrains

Rules still win: the learner only reshapes the probabilities the search sees.

1. Memory: the exact text a person corrected gets that answer next time (one correction is enough).
2. Head: per field, a logistic regression on the base model's probabilities plus hashed text features.
   It is trained in well under a second (pure Python). It is used only while it beats the base model on a
   frozen, human-labelled held-out split (paired bootstrap). Otherwise the base model is used unchanged.

Everything lives in one directory: ``learn.db`` (SQLite: decisions and labels), ``model.json`` (the promoted
heads) and ``versions/`` (the last 10 models, for rollback).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .search import normalize
from .when import same

FORMAT = 1
FEAT_VERSION = 1
MAX_TOKENS = 256
MIN_TRAIN = 5
MIN_TEST = 5
KEEP_VERSIONS = 10
TEST_SOURCES = ("human", "approval")
LAM = 1e-3
LR = 0.5
EPOCHS = 20
BOOTSTRAP = 2000
_FLOOR = 1e-6
_TOKEN = re.compile(r"[\w\-./:@=]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_key(text: str) -> str:
    """sha256 of the text, lowercased with whitespace collapsed."""
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def features(text: str) -> list[int]:
    """Hashed feature ids of a text (version 1, frozen: stored ids must stay valid). Binary, deduplicated."""
    toks = _TOKEN.findall(text.lower())[:MAX_TOKENS]
    names = ["b"] + ["u:" + t for t in toks] + ["g:" + a + " " + b for a, b in zip(toks, toks[1:])]
    return sorted({zlib.crc32(n.encode()) % 2**20 for n in names})


def _opt(o: Any) -> str:
    return json.dumps(o)


def _log_floor(p: float) -> float:
    return math.log(max(p, _FLOOR))


def _softmax(s: list[float]) -> list[float]:
    top = max(s)
    e = [math.exp(x - top) for x in s]
    z = sum(e)
    return [x / z for x in e]


def head_probs(head: dict | None, options: list[str], base: list[float], feats: list[int]) -> list[float]:
    """Probabilities over ``options`` (JSON-encoded) from a head, or from the base alone when head is None."""
    lp = [_log_floor(p) for p in base]
    if head is None:
        return _softmax(lp)
    a, b, w = head["a"], head["b"], head["w"]
    s = [a * lp[k] + b.get(o, 0.0) for k, o in enumerate(options)]
    for j in feats:
        row = w.get(str(j))
        if row:
            for k, o in enumerate(options):
                s[k] += row.get(o, 0.0)
    return _softmax(s)


def train_head(rows: list[dict]) -> dict:
    """Fit a head on rows ``{"options", "base", "feats", "y"}`` (y = index of the label). Returns a, b, w."""
    universe: dict[str, int] = {}
    data = []
    for r in rows:
        idx = [universe.setdefault(o, len(universe)) for o in r["options"]]
        data.append((idx, [_log_floor(p) for p in r["base"]], r["feats"], r["y"]))
    n_opt = len(universe)
    a, g_a = 1.0, 0.0
    b, g_b = [0.0] * n_opt, [0.0] * n_opt
    w: dict[int, list[float]] = {}
    g_w: dict[int, list[float]] = {}
    order = list(range(len(data)))
    import random

    rng = random.Random(0)
    sqrt = math.sqrt
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for i in order:
            idx, lp, feats, y = data[i]
            m = len(idx)
            s = [a * lp[k] + b[idx[k]] for k in range(m)]
            for j in feats:
                row = w.get(j)
                if row is not None:
                    for k in range(m):
                        s[k] += row[idx[k]]
            top = max(s)
            e = [math.exp(x - top) for x in s]
            z = sum(e)
            g = [e[k] / z for k in range(m)]
            g[y] -= 1.0
            ga = sum(g[k] * lp[k] for k in range(m)) + LAM * (a - 1.0)
            g_a += ga * ga
            a -= LR * ga / (sqrt(g_a) + 1e-8)
            for k in range(m):
                o = idx[k]
                gk = g[k] + LAM * b[o]
                g_b[o] += gk * gk
                b[o] -= LR * gk / (sqrt(g_b[o]) + 1e-8)
            for j in feats:
                row = w.get(j)
                if row is None:
                    row = w[j] = [0.0] * n_opt
                    g_w[j] = [0.0] * n_opt
                grow = g_w[j]
                for k in range(m):
                    o = idx[k]
                    gk = g[k] + LAM * row[o]
                    grow[o] += gk * gk
                    row[o] -= LR * gk / (sqrt(grow[o]) + 1e-8)
    names = sorted(universe, key=universe.get)
    return {"a": round(a, 6),
            "b": {o: round(b[universe[o]], 6) for o in names if abs(b[universe[o]]) > 1e-9},
            "w": {str(j): {o: round(row[universe[o]], 6) for o in names if abs(row[universe[o]]) > 1e-9}
                  for j, row in w.items() if any(abs(x) > 1e-9 for x in row)}}


def _evaluate(head: dict | None, rows: list[dict]) -> tuple[float, float, list[float]]:
    """(accuracy, mean NLL, per-example NLL) of a head (None = the base model) on rows."""
    nlls, right = [], 0
    for r in rows:
        q = head_probs(head, r["options"], r["base"], r["feats"])
        nlls.append(-math.log(max(q[r["y"]], 1e-12)))
        right += q.index(max(q)) == r["y"]
    n = len(rows)
    return right / n, sum(nlls) / n, nlls


def _bootstrap(diffs: list[float]) -> tuple[float, float]:
    """95% interval of the mean of ``diffs`` (2,000 resamples)."""
    import random

    rng = random.Random(0)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def _judge(head: dict | None, test: list[dict]) -> dict:
    """Compare a head with the base model on the held-out rows and say whether it passes the gate."""
    n = len(test)
    if n == 0:
        return {"n_test": 0, "base": None, "head": None, "nll_gain_ci": None, "pass": False,
                "why": "no held-out human labels yet"}
    acc_b, nll_b, nb = _evaluate(None, test)
    out: dict[str, Any] = {"n_test": n, "base": {"accuracy": acc_b, "nll": nll_b}, "head": None,
                           "nll_gain_ci": None, "pass": False}
    if head is None:
        return out
    acc_h, nll_h, nh = _evaluate(head, test)
    lo, hi = _bootstrap([x - y for x, y in zip(nb, nh)])
    out["head"] = {"accuracy": acc_h, "nll": nll_h}
    out["nll_gain_ci"] = [lo, hi]
    if n < MIN_TEST:
        out["why"] = f"only {n} held-out human labels; needs {MIN_TEST} to prove the head is better"
    elif len({r["options"][r["y"]] for r in test}) < 2:
        # e.g. only approvals were seen: a test with one answer cannot show the head tells cases apart
        out["why"] = f"all {n} held-out labels give the same answer; needs at least two different answers"
    elif lo <= 0:
        out["why"] = f"not proven better than the base model on {n} held-out labels, so the base model is used"
    elif acc_h < acc_b:
        out["why"] = f"better log-loss but lower accuracy than the base model on {n} held-out labels"
    else:
        out["pass"] = True
        out["why"] = f"better than the base model on {n} held-out labels (log-loss gain {lo:.3f} to {hi:.3f})"
    return out


class Learner:
    """Corrections store, memory and per-field heads for one decision. One directory per learner."""

    def __init__(self, path: str | os.PathLike, *, keep_text: bool = True):
        self.dir = Path(path).expanduser()
        self.keep_text = keep_text
        self.db_path = self.dir / "learn.db"
        self.model_path = self.dir / "model.json"
        self.versions_dir = self.dir / "versions"
        self._ready = False
        self._model: dict = {"version": 0, "heads": {}}
        self._stamp: int | None = None

    # -- storage ------------------------------------------------------------------------------------
    def _connect(self):
        import sqlite3
        if not self._ready:
            self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not self.db_path.exists():
                os.close(os.open(self.db_path, os.O_CREAT | os.O_WRONLY, 0o600))
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        if not self._ready:
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, time TEXT, schema_digest TEXT, "
                "key TEXT, text TEXT NULL, feats TEXT, fields TEXT);"
                "CREATE TABLE IF NOT EXISTS labels (rowid INTEGER PRIMARY KEY AUTOINCREMENT, decision_id TEXT, "
                "field TEXT, label TEXT, source TEXT, split TEXT, time TEXT);"
                "CREATE INDEX IF NOT EXISTS decisions_key ON decisions(key);"
                "CREATE INDEX IF NOT EXISTS labels_decision ON labels(decision_id);"
                "CREATE INDEX IF NOT EXISTS labels_field ON labels(field);")
            conn.commit()
            self._ready = True
        return conn

    @contextlib.contextmanager
    def _db(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _write_json(self, path: Path, body: dict) -> None:
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False))
        os.replace(tmp, path)

    # -- model --------------------------------------------------------------------------------------
    def model(self) -> dict:
        """The current model (promoted heads), reloaded when another process changed the file."""
        try:
            stamp = self.model_path.stat().st_mtime_ns
        except FileNotFoundError:
            self._model, self._stamp = {"version": 0, "heads": {}}, None
            return self._model
        if stamp != self._stamp:
            body = json.loads(self.model_path.read_text())
            if not isinstance(body, dict) or body.get("shishya_learner") != FORMAT:
                raise ValueError(f"{self.model_path} is not a Shishya learner model")
            self._model, self._stamp = body, stamp
        return self._model

    @property
    def version(self) -> int:
        return int(self.model().get("version", 0))

    def wrap(self, backend: Any) -> LearnedBackend:
        """The backend, with this learner's head and memory applied to its probabilities."""
        return LearnedBackend(self, backend)

    # -- logging and corrections --------------------------------------------------------------------
    def log(self, decision_id: str, schema_digest: str, text: str, last: dict) -> None:
        """Store what the model was asked and what it said (base probabilities), so a later correction can find it."""
        fields = {n: {"options": last["options"][n], "base": last["base"][n]} for n in last["base"]}
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (decision_id, _now(), schema_digest, last["key"], text if self.keep_text else None,
                        json.dumps(last["feats"]), json.dumps(fields)))

    def correct(self, decision_id: str, answer: dict, *, source: str = "human", train: bool = True) -> dict:
        """Record the right answer for fields of a logged decision and retrain those fields' heads."""
        if not isinstance(answer, dict):
            raise ValueError("the answer must be an object: {field: value}")
        with self._db() as db:
            row = db.execute("SELECT key, fields FROM decisions WHERE id = ?", (decision_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown decision id {decision_id!r}")
            key, fields = row[0], json.loads(row[1])
            test_ok = int(key[:8], 16) % 5 == 0 and source in TEST_SOURCES
            recorded, skipped = [], {}
            for name, value in answer.items():
                if name not in fields:
                    skipped[name] = "the model was not asked this field in that decision"
                    continue
                if not any(same(o, value) for o in fields[name]["options"]):
                    skipped[name] = f"not one of the options that were allowed then: {fields[name]['options']}"
                    continue
                db.execute("INSERT INTO labels (decision_id, field, label, source, split, time) "
                           "VALUES (?, ?, ?, ?, ?, ?)",
                           (decision_id, name, json.dumps(value), source, "test" if test_ok else "train", _now()))
                recorded.append(name)
        report = self.train(recorded) if train and recorded else None
        return {"recorded": recorded, "skipped": skipped, "train": report}

    # -- training -----------------------------------------------------------------------------------
    def _rows(self, db, name: str) -> dict[str, list[dict]]:
        """Newest label per decision for a field, as training rows, split into train and test."""
        latest: dict[str, tuple] = {}
        for did, label, split, feats, fields in db.execute(
                "SELECT l.decision_id, l.label, l.split, d.feats, d.fields FROM labels l "
                "JOIN decisions d ON d.id = l.decision_id WHERE l.field = ? ORDER BY l.rowid", (name,)):
            latest[did] = (label, split, feats, fields)
        out: dict[str, list[dict]] = {"train": [], "test": []}
        for label, split, feats, fields in latest.values():
            f = json.loads(fields)[name]
            options = [_opt(o) for o in f["options"]]
            if label not in options:
                continue
            out[split].append({"options": options, "base": [p for p in f["base"]], "feats": json.loads(feats),
                               "y": options.index(label)})
        return out

    def train(self, fields: list[str] | None = None) -> dict:
        """Retrain heads and run the promotion gate. Returns a report per field."""
        t0 = time.perf_counter()
        model = self.model()
        heads = {n: dict(h) for n, h in model.get("heads", {}).items()}
        before = {n: {k: h[k] for k in ("a", "b", "w")} for n, h in heads.items()}
        with self._db() as db:
            if fields is None:
                fields = [r[0] for r in db.execute("SELECT DISTINCT field FROM labels ORDER BY field")]
            rows = {n: self._rows(db, n) for n in fields}
        reports = {}
        for name in fields:
            reports[name] = self._train_field(name, rows[name]["train"], rows[name]["test"], heads)
        after = {n: {k: h[k] for k in ("a", "b", "w")} for n, h in heads.items()}
        changed = after != before
        version = int(model.get("version", 0))
        if changed:
            version = self._save_model(version + 1, heads)
        out = {"fields": reports, "version": version, "changed": changed,
               "train_ms": round((time.perf_counter() - t0) * 1000, 1)}
        self._write_json(self.dir / "last_report.json", out)
        return out

    def _train_field(self, name: str, train: list[dict], test: list[dict], heads: dict) -> dict:
        t0 = time.perf_counter()
        rep: dict[str, Any] = {"n_train": len(train), "n_test": len(test), "base": None, "head": None,
                               "nll_gain_ci": None, "promoted": False, "kept_previous": False}
        if len(train) < MIN_TRAIN:
            heads.pop(name, None)
            rep["why"] = (f"needs {MIN_TRAIN - len(train)} more labels (has {len(train)}, "
                          f"needs {MIN_TRAIN} to train a head)")
            rep["train_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return rep
        cand = train_head(train)
        verdict = _judge(cand, test)
        rep["train_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if verdict["pass"]:
            heads[name] = {**cand, "report": {k: verdict[k] for k in ("n_test", "base", "head", "nll_gain_ci")}}
            rep.update(base=verdict["base"], head=verdict["head"], nll_gain_ci=verdict["nll_gain_ci"],
                       promoted=True, why=verdict["why"])
            return rep
        rep.update(base=verdict["base"], head=verdict["head"], nll_gain_ci=verdict["nll_gain_ci"],
                   why=verdict["why"])
        prev = heads.get(name)
        if prev is not None:
            still = _judge(prev, test)
            if still["pass"]:
                prev["report"] = {k: still[k] for k in ("n_test", "base", "head", "nll_gain_ci")}
                rep.update(base=still["base"], head=still["head"], nll_gain_ci=still["nll_gain_ci"],
                           promoted=True, kept_previous=True,
                           why="the new head was not better, so the previous head is kept: " + still["why"])
                return rep
            heads.pop(name)
        return rep

    def _save_model(self, version: int, heads: dict, **extra: Any) -> int:
        body = {"shishya_learner": FORMAT, "version": version, "feat": FEAT_VERSION, "heads": heads,
                "created": _now(), **extra}
        self.versions_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._write_json(self.versions_dir / f"model-{version}.json", body)
        self._write_json(self.model_path, body)
        for old in sorted(self._version_files())[:-KEEP_VERSIONS]:
            (self.versions_dir / f"model-{old}.json").unlink(missing_ok=True)
        return version

    def _version_files(self) -> list[int]:
        if not self.versions_dir.exists():
            return []
        found = (re.fullmatch(r"model-(\d+)\.json", p.name) for p in self.versions_dir.iterdir())
        return [int(m.group(1)) for m in found if m]

    def rollback(self) -> int:
        """Bring back the heads of the previous model version, saved as a NEW version (receipts stay unambiguous).

        Returns the new version number."""
        current = self.version
        restored = self.model().get("rolled_back_to") or current     # rolling back twice goes further back
        earlier = [v for v in self._version_files() if v < restored]
        if not earlier:
            raise ValueError("there is no earlier model version to go back to")
        prev = json.loads((self.versions_dir / f"model-{max(earlier)}.json").read_text())
        new = max(self._version_files() + [current]) + 1
        return self._save_model(new, prev.get("heads", {}), rolled_back_to=prev.get("version"))

    # -- status -------------------------------------------------------------------------------------
    def status(self) -> dict:
        """Label counts per field, the model version, which heads are promoted and the last train report."""
        with self._db() as db:
            fields: dict[str, dict] = {}
            for name, split, n in db.execute("SELECT field, split, COUNT(*) FROM labels GROUP BY field, split"):
                fields.setdefault(name, {"train": 0, "test": 0, "memory": 0})[split] = n
            for name, n in db.execute(
                    "SELECT l.field, COUNT(DISTINCT d.key) FROM labels l JOIN decisions d ON d.id = l.decision_id "
                    "GROUP BY l.field"):
                fields[name]["memory"] = n
            n_decisions = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        model = self.model()
        for name in model.get("heads", {}):
            fields.setdefault(name, {"train": 0, "test": 0, "memory": 0})
        for name, f in fields.items():
            f["promoted"] = name in model.get("heads", {})
        try:
            last = json.loads((self.dir / "last_report.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            last = None
        return {"version": int(model.get("version", 0)), "decisions": n_decisions, "fields": fields,
                "last_report": last}

    # -- memory -------------------------------------------------------------------------------------
    def memory(self, key: str, options: dict[str, list[str]]) -> dict[str, tuple[str, int]]:
        """Per field, the newest label for this text (JSON) and how many newest labels in a row agree."""
        found: dict[str, list[str]] = {}
        with self._db() as db:
            for fields, name, label in db.execute(
                    "SELECT d.fields, l.field, l.label FROM labels l JOIN decisions d ON d.id = l.decision_id "
                    "WHERE d.key = ? ORDER BY l.rowid DESC", (key,)):
                if name in options and [_opt(o) for o in json.loads(fields)[name]["options"]] == options[name]:
                    found.setdefault(name, []).append(label)
        out = {}
        for name, labels in found.items():
            run = 0
            for x in labels:
                if x != labels[0]:
                    break
                run += 1
            out[name] = (labels[0], run)
        return out


class LearnedBackend:
    """Wrap any backend: apply the promoted head, then memory. ``last`` keeps what the engine needs to log."""

    name = "learned"

    def __init__(self, learner: Learner, backend: Any):
        self.learner = learner
        self.backend = backend
        self.last: dict = {}

    @property
    def model(self) -> str:
        return f"{getattr(self.backend, 'model', self.backend.name)} + learner v{self.learner.version}"

    def score(self, text: str, fields: list, options: dict[str, tuple]) -> dict[str, dict[Any, float]]:
        out = self.backend.score(text, fields, options)
        key, feats = text_key(text), features(text)
        base = {n: [normalize(p).get(o, 0.0) for o in options[n]] for n, p in out.items()}
        self.last = {"base": base, "used": dict.fromkeys(out), "feats": feats, "key": key, "version": 0,
                     "options": {n: list(options[n]) for n in out}}
        try:
            return self._learned(out, options, base, key, feats)
        except Exception as e:  # noqa: BLE001 - a broken store or model falls back to the base model
            self.last["error"] = f"{type(e).__name__}: {e}"
            return {n: dict(zip(options[n], base[n])) for n in out}

    def _learned(self, out: dict, options: dict, base: dict, key: str, feats: list[int]) -> dict:
        model = self.learner.model()
        enc = {n: [_opt(o) for o in options[n]] for n in out}
        memory = self.learner.memory(key, enc) if out else {}
        used, result = {}, {}
        for name in out:
            opts = list(options[name])
            b = base[name]
            q, used[name] = b, None
            head = model.get("heads", {}).get(name)
            if head is not None:
                q, used[name] = head_probs(head, enc[name], b, feats), "head"
            if name in memory:
                label, run = memory[name]
                w = run / (run + 0.25)
                q = [(1 - w) * x + (w if enc[name][k] == label else 0.0) for k, x in enumerate(q)]
                used[name] = "memory"
            result[name] = dict(zip(opts, q))
        self.last.update(used=used, version=int(model.get("version", 0)))
        return result
