#!/usr/bin/env bash
# Run the full life test on Kaggle's free GPU with your own account.
#
#   export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
#   brain/kaggle/run.sh YOUR_KAGGLE_USERNAME
#
# 1. uploads brain/ (code only) as a PRIVATE Kaggle dataset <user>/shishya-brain-code
# 2. pushes a private GPU kernel <user>/shishya-brain-life: GPT, BDH and the Brain on the same 100M bytes,
#    then the life test
# 3. waits, and downloads the results to brain/out/kaggle-<kernel>/  (--canary: a 15-minute GPU check first)
set -euo pipefail
user=${1:?usage: run.sh KAGGLE_USERNAME [--canary]}
canary=${2:-}
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
kaggle="${KAGGLE:-kaggle}"
out="$root/brain/out"

ds="$out/kaggle_code"
rm -rf "$ds" && mkdir -p "$ds"
tar -czf "$ds/brain.tar.gz" -C "$root" --exclude=brain/out --exclude=__pycache__ brain
git -C "$root" rev-parse HEAD > "$ds/commit.txt"
printf '{"title": "shishya-brain-code", "id": "%s/shishya-brain-code", "licenses": [{"name": "other"}]}\n' "$user" \
  > "$ds/dataset-metadata.json"
if "$kaggle" datasets status "$user/shishya-brain-code" >/dev/null 2>&1; then
  "$kaggle" datasets version -p "$ds" -m "$(cat "$ds/commit.txt")"
else
  "$kaggle" datasets create -p "$ds"
fi
for _ in $(seq 1 60); do
  [ "$("$kaggle" datasets status "$user/shishya-brain-code" 2>/dev/null | tail -1)" = "ready" ] && break
  sleep 10
done

slug="shishya-brain-life"
kdir="$out/kernel"
[ "$canary" = "--canary" ] && slug="shishya-brain-canary" && kdir="$out/kernel-canary"
rm -rf "$kdir" && mkdir -p "$kdir"
if [ "$canary" = "--canary" ]; then
  sed 's/^CANARY = False/CANARY = True/' "$here/kernel.py" > "$kdir/kernel.py"
else
  cp "$here/kernel.py" "$kdir/kernel.py"
fi
printf '{"id": "%s/%s", "title": "%s", "code_file": "kernel.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": ["%s/shishya-brain-code"], "competition_sources": [], "kernel_sources": []}\n' \
  "$user" "$slug" "$slug" "$user" > "$kdir/kernel-metadata.json"
"$kaggle" kernels push -p "$kdir"

echo "running on Kaggle (a few hours)..."
for _ in $(seq 1 600); do
  s=$("$kaggle" kernels status "$user/$slug" 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $s"
  case "$s" in *COMPLETE*) break ;; *ERROR*|*CANCEL*) "$kaggle" kernels output "$user/$slug" -p "$out/kaggle-$slug" || true; exit 1 ;; esac
  sleep 60
done
mkdir -p "$out/kaggle-$slug"
"$kaggle" kernels output "$user/$slug" -p "$out/kaggle-$slug"
echo "results: $out/kaggle-$slug/out/life.json"
