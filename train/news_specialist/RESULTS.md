# News specialist: how it was trained and how it does

**Task:** read one financial news headline about a company and say whether it is **bearish, bullish or neutral** for that company. In Shishya this answers one field (`news_impact`) while a general decision model answers the rest, and the rules sit on top.

## Data (all human-labelled, all MIT)

| Source | What | Used for |
|---|---|---|
| [SEntFiN 1.0](https://github.com/pyRis/SEntFiN) | 10,753 Indian business headlines (2011–2015), sentiment per company | train / dev / **Indian test** |
| [twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 11,931 US financial news headlines | train / dev / **US test** |

- Every SEntFiN (headline, company) pair becomes `COMPANY: headline`, the same shape as an exchange alert.
- SEntFiN is split **by headline text**, 80/10/10. The file repeats some headlines under new row numbers, and a split by row number would have leaked them into the test set. A check in `prepare_data.py` fails the build if any headline appears in two splits.
- No label comes from an AI model.

## Training

- **Recipe:** `train.py` on Kaggle's free Tesla T4, ModernBERT base (149M) and large (395M), 4 epochs, lr 5e-5, batch 32.
- **Choices made on dev only:** the checkpoint (best dev macro-F1) and a temperature for calibration.
- **Tests:** reported once and never used for any choice.
- **Shipped choice:** large had the higher dev macro-F1 (0.895 vs 0.883). Both are published, because base is 2.6× smaller and 2× faster for −0.6 points.

| | base | large |
|---|---:|---:|
| Training time (free T4) | 13 min | 32 min |
| Size (fp16) | 293 MB | 772 MB |
| Laptop speed (Apple M2, MPS) | 17 ms | 35 ms |
| Dev macro-F1 | 0.883 | 0.895 |
| **Indian test** (1,438): accuracy / macro-F1 / ECE | 88.8% / 0.889 / 0.014 | 89.6% / 0.895 / 0.022 |
| **US test** (2,388): accuracy / macro-F1 / ECE | 88.9% / 0.854 / 0.012 | 89.9% / 0.868 / 0.012 |

## Against other models, on the same held-out headlines

| Indian headlines, same random 300 | Accuracy | Macro-F1 | Time each |
|---|---:|---:|---:|
| Kev-0.8B (general decision model, local) | 56.3% | 0.499 | 93 ms |
| [FinBERT](https://huggingface.co/ProsusAI/finbert) (2019 financial-sentiment model, local) | 62.3% | 0.637 | 22 ms |
| Nemotron 3.5 Super (free big LLM, 20 per request) | 78.3% | 0.783 | 1,571 ms |
| Nemotron 3.5 Super (one headline per request) | 73.7% | 0.736 | 3,737 ms |
| **Shishya news specialist, base** | **87.7%** | **0.880** | **17 ms** |
| Shishya news specialist, large | 86.3% | 0.866 | 35 ms |

| US headlines, same random 300 | Accuracy | Macro-F1 |
|---|---:|---:|
| Kev-0.8B | 73.7% | 0.597 |
| FinBERT | 75.7% | 0.704 |
| Nemotron 3.5 Super (20 per request) | 73.3% | 0.733 |
| **Shishya news specialist (stage 1, US data only)** | **87.0%** | **0.837** |

Asking the LLM one headline at a time did **not** help it: 0.736 vs 0.783 when it saw 20 headlines per request, on the
same 300 headlines (they agree on 88% of them). The README chart uses its better score.

## What didn't work, and the blind spot

- **Stage 1 (US data only) didn't transfer to India.** On our hand-written Indian exchange-alert headlines it scored 18/33, below Kev's 22/33. It called "Promoter pledge rises to 78%" bullish. That's why stage 2 adds SEntFiN.
- **Exchange-filing language is still a blind spot.** Stage 2 scores 21/33 (base) and 20/33 (large) on those same 11 alert templates (× 3 company names), against Kev's 22/33. It misses red flags that need filing know-how:
  - an auditor resigning "citing non-cooperation"
  - promoter pledges rising
  - and on the good side, an order worth 1.4× revenue
- **On the stock-alert part of rulebench it changed 4 of 20 `news_impact` calls but no final actions.** Most of that scenario's errors come from the general model's `hold`/`watch` choice. Results unchanged at 12/20, so we don't claim a benchmark gain there.
- **Next (v0.3):** add real NSE filings with checked labels, and test on held-out filings.

## Reproduce

```bash
export KAGGLE_API_TOKEN=...       # your token
train/news_specialist/run_kaggle.sh YOUR_KAGGLE_USERNAME base
python train/news_specialist/baselines.py kev --split india            # needs a decision server on :8009
python train/news_specialist/baselines.py llm --split india --n 300    # needs OPENROUTER_API_KEY
```
