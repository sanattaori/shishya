"""Baselines for the news specialist, on the same human-labelled test split (2,388 headlines).

  kev       a System One decision server (e.g. Kev-0.8B), asked the 3-way question per headline
  llm       an OpenRouter chat model, 20 headlines per request, on a fixed random sample

    python train/news_specialist/baselines.py kev --url http://127.0.0.1:8009
    python train/news_specialist/baselines.py llm --n 300

Writes train/news_specialist/out/<name>.json with accuracy, macro-F1 and every prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
LABELS = ["bearish", "bullish", "neutral"]
QUESTION = "For the company or stock in this headline, is the news bearish, bullish or neutral?"
CRITERIA = {"bearish": "Bad news: likely to push the stock down",
            "bullish": "Good news: likely to push the stock up",
            "neutral": "No clear effect on the stock"}


def test_split(name: str = "us") -> list[dict]:
    if name != "us":
        return [json.loads(x) for x in (HERE / "out" / "data" / f"test_{name}.jsonl").read_text().splitlines() if x]
    cache = HERE / "out" / "test.jsonl"
    if cache.exists():
        return [json.loads(x) for x in cache.read_text().splitlines()]
    rows, off = [], 0
    with httpx.Client(timeout=60) as c:
        while True:
            r = c.get("https://datasets-server.huggingface.co/rows", params={
                "dataset": "zeroshot/twitter-financial-news-sentiment", "config": "default",
                "split": "validation", "offset": off, "length": 100})
            r.raise_for_status()
            batch = [x["row"] for x in r.json()["rows"]]
            rows += [{"text": b["text"], "label": LABELS[b["label"]]} for b in batch]
            off += len(batch)
            if len(batch) < 100:
                break
    cache.parent.mkdir(exist_ok=True)
    cache.write_text("\n".join(json.dumps(x) for x in rows))
    return rows


def score(preds: list[str], gold: list[str]) -> dict:
    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    f1s = []
    for c in LABELS:
        tp = sum(p == c and g == c for p, g in zip(preds, gold))
        fp = sum(p == c and g != c for p, g in zip(preds, gold))
        fn = sum(p != c and g == c for p, g in zip(preds, gold))
        pr, rc = (tp / (tp + fp) if tp + fp else 0), (tp / (tp + fn) if tp + fn else 0)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return {"n": len(gold), "accuracy": acc, "macro_f1": sum(f1s) / 3}


def run_kev(rows: list[dict], url: str) -> dict:
    preds, t0 = [], time.perf_counter()
    with httpx.Client(timeout=120) as c:
        for i, r in enumerate(rows):
            body = {"state": r["text"], "questions": {"s": {"type": "choice", "instructions": QUESTION,
                                                             "criteria": CRITERIA}}}
            a = c.post(url.rstrip("/") + "/v1/systemone", json=body).json()["answers"]["s"]
            preds.append(max(a["probabilities"], key=a["probabilities"].get))
            if (i + 1) % 200 == 0:
                print(f"kev {i + 1}/{len(rows)}", file=sys.stderr)
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    return {"preds": preds, "ms_per_item": ms}


def run_finbert(rows: list[dict], model: str = "ProsusAI/finbert") -> dict:
    """FinBERT (2019): positive / negative / neutral, mapped to bullish / bearish / neutral."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(model).eval()
    names = {i: n.lower() for i, n in net.config.id2label.items()}
    to = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
    preds, t0 = [], time.perf_counter()
    with torch.no_grad():
        for r in rows:
            logits = net(**tok(r["text"], truncation=True, max_length=128, return_tensors="pt")).logits[0]
            preds.append(to[names[int(logits.argmax())]])
    return {"preds": preds, "ms_per_item": (time.perf_counter() - t0) * 1000 / len(rows)}


def run_llm(rows: list[dict], model: str, per_request: int = 20) -> dict:
    key = os.environ["OPENROUTER_API_KEY"]
    preds: list[str | None] = [None] * len(rows)
    t0 = time.perf_counter()
    with httpx.Client(timeout=180) as c:
        for start in range(0, len(rows), per_request):
            chunk = rows[start:start + per_request]
            listing = "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(chunk))
            prompt = (f"{QUESTION}\nAnswer for each numbered headline. Reply with JSON only: "
                      f'{{"answers": ["bearish" | "bullish" | "neutral", ...]}} in the same order.\n\n{listing}')
            for attempt in range(4):
                r = c.post("https://openrouter.ai/api/v1/chat/completions",
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                                 "messages": [{"role": "user", "content": prompt}]})
                if r.status_code == 200:
                    try:
                        txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", r.json()["choices"][0]["message"]["content"])
                        ans = json.loads(txt)["answers"]
                        if len(ans) == len(chunk):
                            for i, a in enumerate(ans):
                                preds[start + i] = a if a in LABELS else None
                            break
                    except (KeyError, ValueError, TypeError):
                        pass
                time.sleep(3 * (attempt + 1))
            if (start // per_request) % max(1, 20 // per_request) == 0:
                print(f"llm {min(start + per_request, len(rows))}/{len(rows)}", file=sys.stderr)
    ms = (time.perf_counter() - t0) * 1000 / len(rows)
    return {"preds": preds, "ms_per_item": ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["kev", "llm", "finbert"])
    ap.add_argument("--per-request", type=int, default=20, help="llm: headlines per request (1 = one at a time)")
    ap.add_argument("--url", default="http://127.0.0.1:8009")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-super-vl:free")
    ap.add_argument("--n", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--split", default="us", help="us (US headlines) or india (SEntFiN test)")
    a = ap.parse_args()
    rows = test_split(a.split)
    idx = list(range(len(rows)))
    if a.n:
        idx = sorted(random.Random(0).sample(idx, a.n))
    sub = [rows[i] for i in idx]
    if a.which == "kev":
        res = run_kev(sub, a.url)
    elif a.which == "finbert":
        res = run_finbert(sub)
    else:
        res = run_llm(sub, a.model, a.per_request)
    preds = [p or "none" for p in res["preds"]]
    model = {"llm": a.model, "finbert": "ProsusAI/finbert"}.get(a.which, a.url)
    out = {"which": a.which, "split": a.split, "model": model, "per_request": a.per_request, "idx": idx,
           "ms_per_item": res["ms_per_item"], **score(preds, [r["label"] for r in sub]),
           "unanswered": sum(p == "none" for p in preds), "preds": preds}
    per = f"-per{a.per_request}" if a.which == "llm" and a.per_request != 20 else ""
    name = f"{a.which}{'-' + a.split if a.split != 'us' else ''}{'-' + str(a.n) if a.n else ''}{per}"
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("preds", "idx")}, indent=1))


if __name__ == "__main__":
    main()
