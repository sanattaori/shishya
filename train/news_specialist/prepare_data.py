"""Build the stage-2 training data for the news specialist (all labels written by humans).

Sources (both MIT):
  - SEntFiN 1.0 (github.com/pyRis/SEntFiN): 10,753 Indian business headlines, sentiment per company
  - zeroshot/twitter-financial-news-sentiment: 9,543 + 2,388 US financial news headlines

Each SEntFiN (headline, company) pair becomes one example "COMPANY: headline", the same shape as
an exchange alert ("ACMEIND: Statutory auditor resigns ..."). SEntFiN is split by HEADLINE TEXT (80/10/10,
seeded; the file repeats some headlines under new row numbers), so no headline is in two splits. The US validation split stays the US test.

    python train/news_specialist/prepare_data.py      # writes train/news_specialist/out/data/*.jsonl
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "data"
SENTFIN_URL = "https://raw.githubusercontent.com/pyRis/SEntFiN/main/SEntFiN.csv"
MAP = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}
PAIR = re.compile(r"[{,]\s*['\"](.+?)['\"]:\s*'(positive|negative|neutral)'")
SEED = 0

sys.path.insert(0, str(HERE))
from baselines import LABELS, test_split  # noqa: E402


def sentfin() -> list[dict]:
    path = OUT / "SEntFiN.csv"
    if not path.exists():
        import httpx
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(httpx.get(SENTFIN_URL, timeout=60, follow_redirects=True).content)
    rows, seen = [], set()
    for r in csv.DictReader(path.open(encoding="utf-8")):
        title = " ".join(r["Title"].split())
        hid = title.lower()                  # the dataset repeats some headlines under new row numbers
        for entity, label in PAIR.findall(r["Decisions"]):
            key = (hid, entity.lower(), label)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"hid": hid, "text": f"{entity}: {title}", "label": MAP[label], "source": "sentfin"})
    return rows


def us_train() -> list[dict]:
    path = OUT / "sent_train.csv"
    if not path.exists():
        import httpx
        url = "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment/resolve/main/sent_train.csv"
        path.write_bytes(httpx.get(url, timeout=120, follow_redirects=True).content)
    return [{"text": r["text"], "label": LABELS[int(r["label"])], "source": "us"}
            for r in csv.DictReader(path.open(encoding="utf-8"))]


def main() -> None:
    rng = random.Random(SEED)
    sf = sentfin()
    hids = sorted({r["hid"] for r in sf})
    rng.shuffle(hids)
    n = len(hids)
    part = {h: ("test" if i < n // 10 else "dev" if i < n // 5 else "train") for i, h in enumerate(hids)}
    us = us_train()
    rng.shuffle(us)
    us_dev, us_tr = us[:len(us) // 10], us[len(us) // 10:]
    splits = {
        "train": [r for r in sf if part[r["hid"]] == "train"] + us_tr,
        "dev": [r for r in sf if part[r["hid"]] == "dev"] + us_dev,
        "test_india": [r for r in sf if part[r["hid"]] == "test"],
        "test_us": [{**r, "source": "us"} for r in test_split()],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows) if name == "train" else None
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        counts = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
        print(f"{name:<11} {len(rows):>6}  {counts}")
    # no headline may appear in two splits
    seen = {name: {r["hid"] for r in rows if r["source"] == "sentfin"} for name, rows in splits.items()}
    assert not (seen["train"] & seen["test_india"]) and not (seen["dev"] & seen["test_india"]), "leak"
    print("no SEntFiN headline is shared between train/dev and test_india")


if __name__ == "__main__":
    main()
