#!/usr/bin/env bash
# Train the news specialist on Kaggle's free GPU with your own account.
#
#   export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
#   train/news_specialist/run_kaggle.sh YOUR_KAGGLE_USERNAME base     # or: large
#
# 1. builds the human-labelled splits (prepare_data.py)
# 2. uploads them as a PRIVATE Kaggle dataset <user>/shishya-news-data
# 3. pushes a private GPU kernel <user>/shishya-news-<size> running train.py
# 4. waits, downloads the model and metrics to train/news_specialist/out/shishya-news-<size>/
set -euo pipefail
user=${1:?usage: run_kaggle.sh KAGGLE_USERNAME [base|large]}
size=${2:-base}
case "$size" in base|large) ;; *) echo "size must be base or large" >&2; exit 2 ;; esac
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
py="${PYTHON:-$root/.venv/bin/python}"
kaggle="${KAGGLE:-kaggle}"
out="$here/out"

"$py" "$here/prepare_data.py"

ds="$out/kaggle_ds"
mkdir -p "$ds"
cp "$out/data/"{train,dev,test_india,test_us}.jsonl "$ds/"
printf '{"title": "shishya-news-data", "id": "%s/shishya-news-data", "licenses": [{"name": "other"}]}\n' "$user" \
  > "$ds/dataset-metadata.json"
if [ "$("$kaggle" datasets status "$user/shishya-news-data" 2>/dev/null | tail -1)" = "ready" ]; then
  "$kaggle" datasets version -p "$ds" -m "update"
else
  "$kaggle" datasets create -p "$ds"
fi
for _ in $(seq 1 30); do
  [ "$("$kaggle" datasets status "$user/shishya-news-data" 2>/dev/null | tail -1)" = "ready" ] && break
  sleep 10
done

slug="shishya-news-$size"
kdir="$out/kernel-$size"
mkdir -p "$kdir"
sed "s|answerdotai/ModernBERT-base\")|answerdotai/ModernBERT-$size\")|" "$here/train.py" > "$kdir/train.py"
printf '{"id": "%s/%s", "title": "%s", "code_file": "train.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": ["%s/shishya-news-data"], "competition_sources": [], "kernel_sources": []}\n' \
  "$user" "$slug" "$slug" "$user" > "$kdir/kernel-metadata.json"
"$kaggle" kernels push -p "$kdir"

echo "training on Kaggle (about 13 min for base, 32 min for large)..."
for _ in $(seq 1 120); do
  s=$("$kaggle" kernels status "$user/$slug" 2>&1 | tail -1)
  case "$s" in *COMPLETE*) break ;; *ERROR*|*CANCEL*) echo "$s" >&2; exit 1 ;; esac
  sleep 30
done
dest="$out/$slug"
mkdir -p "$dest"
"$kaggle" kernels output "$user/$slug" -p "$dest"
echo "model: $dest/model   metrics: $dest/metrics.json"
