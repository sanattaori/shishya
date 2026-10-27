"""Train a small financial-news specialist on Kaggle's free GPU (runs as a Kaggle script kernel).

Stage 2 data (all human-labelled, MIT), built by prepare_data.py and attached as a Kaggle dataset:
  train.jsonl       SEntFiN (Indian headlines, "COMPANY: headline") + US financial news headlines
  dev.jsonl         used to pick the checkpoint (macro-F1) and fit the temperature; nothing else
  test_*.jsonl      reported only, never used for any choice

Without the attached dataset it falls back to stage 1 (US headlines from the Hugging Face Hub).
Set BASE_MODEL in the kernel's environment to change the base (default ModernBERT-base).

Outputs in /kaggle/working: model/ (fp16 safetensors + tokenizer), metrics.json, predictions_<test>.jsonl
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

BASE = os.environ.get("BASE_MODEL", "answerdotai/ModernBERT-base")
LABELS = ["bearish", "bullish", "neutral"]
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "out"
SEED = 0
EPOCHS, BATCH, LR = 4, 32, 5e-5

try:
    import datasets  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
import transformers

if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 48):     # ModernBERT needs >= 4.48
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.48"])
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
from datasets import Dataset, load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

set_seed(SEED)
t0 = time.time()


def read_jsonl(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    return Dataset.from_list([{"text": r["text"], "label": LABELS.index(r["label"])} for r in rows])


found = sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
if found:
    ddir = os.path.dirname(found[0])
    train, dev = read_jsonl(f"{ddir}/train.jsonl"), read_jsonl(f"{ddir}/dev.jsonl")
    tests = {os.path.basename(p)[5:-6]: read_jsonl(p) for p in sorted(glob.glob(f"{ddir}/test_*.jsonl"))}
    stage = 2
else:
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    split = ds["train"].train_test_split(test_size=0.1, seed=SEED)
    train, dev, tests = split["train"], split["test"], {"us": ds["validation"]}
    stage = 1
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"stage {stage}: train {len(train)} dev {len(dev)} tests { {k: len(v) for k, v in tests.items()} } "
      f"base {BASE} gpu {gpu}")

tok = AutoTokenizer.from_pretrained(BASE)


def enc(b):
    return tok(b["text"], truncation=True, max_length=128)


train_t, dev_t = train.map(enc, batched=True), dev.map(enc, batched=True)
tests_t = {k: v.map(enc, batched=True) for k, v in tests.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)})


def macro_f1(pred, y):
    f1s = []
    for i in range(3):
        tp, fp, fn = ((pred == i) & (y == i)).sum(), ((pred == i) & (y != i)).sum(), ((pred != i) & (y == i)).sum()
        p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), f1s


def metrics_fn(p):
    pred = p.predictions.argmax(-1)
    return {"accuracy": float((pred == p.label_ids).mean()), "macro_f1": macro_f1(pred, p.label_ids)[0]}


use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
warmup = int(0.1 * EPOCHS * -(-len(train_t) // BATCH))     # 10% of the steps (works on transformers 4 and 5)
args = TrainingArguments(
    output_dir=f"{OUT}/ckpt", num_train_epochs=EPOCHS, learning_rate=LR, per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=128, warmup_steps=warmup, weight_decay=0.01, lr_scheduler_type="linear",
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="macro_f1", bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
    report_to="none", seed=SEED, logging_steps=100)
trainer = Trainer(model=model, args=args, train_dataset=train_t, eval_dataset=dev_t,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics_fn)
trainer.train()


def logits_of(d):
    return trainer.predict(d).predictions.astype(np.float64)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nll(T, z, y):
    return float(-np.log(softmax(z / T)[np.arange(len(y)), y] + 1e-12).mean())


def ece(p, y, bins=10):
    conf, pred = p.max(-1), p.argmax(-1)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any()))


dev_z, dev_y = logits_of(dev_t), np.array(dev["label"])
Ts = np.exp(np.linspace(np.log(0.25), np.log(8), 400))
T = float(Ts[np.argmin([nll(t, dev_z, dev_y) for t in Ts])])
dev_pred = softmax(dev_z / T).argmax(-1)

metrics = {"stage": stage, "base": BASE, "train": len(train), "dev": len(dev), "temperature": T,
           "epochs": EPOCHS, "lr": LR, "batch": BATCH, "gpu": gpu,
           "dev_macro_f1": macro_f1(dev_pred, dev_y)[0], "dev_accuracy": float((dev_pred == dev_y).mean()),
           "tests": {}}
for name, d in tests_t.items():
    z, y = logits_of(d), np.array(tests[name]["label"])
    p = softmax(z / T)
    pred = p.argmax(-1)
    mf1, f1s = macro_f1(pred, y)
    metrics["tests"][name] = {
        "n": len(y), "accuracy": float((pred == y).mean()), "macro_f1": mf1,
        "f1": dict(zip(LABELS, map(float, f1s))), "ece_raw": ece(softmax(z), y), "ece_calibrated": ece(p, y),
        "confusion": [[int(((y == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)]}
    with open(f"{OUT}/predictions_{name}.jsonl", "w") as f:
        for text, yy, pp in zip(tests[name]["text"], y, p):
            f.write(json.dumps({"text": text, "label": LABELS[yy], "probs": dict(zip(LABELS, map(float, pp)))}) + "\n")
metrics["minutes"] = (time.time() - t0) / 60
print(json.dumps(metrics, indent=1))

model.config.shishya_temperature = T
model.half().save_pretrained(f"{OUT}/model", safe_serialization=True)
tok.save_pretrained(f"{OUT}/model")
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
shutil.rmtree(f"{OUT}/ckpt", ignore_errors=True)     # keep only the final model in the outputs
print("done")
