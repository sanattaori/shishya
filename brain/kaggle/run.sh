#!/usr/bin/env bash
# Run the life test on Kaggle's free GPU with your own account.
#
#   export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
#   brain/kaggle/run.sh YOUR_KAGGLE_USERNAME [--canary | --brain2 | --life2]
#
# (no flag)  GPT, BDH and the v1 Brain on the same 100M bytes, then the life test   kernel shishya-brain-life
# --canary   the same, 20 steps each and a short life: GPU-only bugs in minutes     kernel shishya-brain-canary
# --brain2   train the v2 Brain only                                                 kernel shishya-brain-v2
# --life2    life test of v2 against the first run's GPT and BDH (needs both done)  kernel shishya-brain-life2
#
# Uploads brain/ (code only) as a PRIVATE dataset <user>/shishya-brain-code, pushes a private GPU kernel, waits,
# and downloads its output to brain/out/kaggle-<kernel>/.
set -euo pipefail
user=${1:?usage: run.sh KAGGLE_USERNAME [--canary | --brain2 | --life2]}
flag=${2:-}
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
kaggle="${KAGGLE:-kaggle}"
out="$root/brain/out"
sources=""
case "$flag" in
  "") mode=full; slug=shishya-brain-life ;;
  --canary) mode=canary; slug=shishya-brain-canary ;;
  --brain2) mode=brain2; slug=shishya-brain-v2 ;;
  --life2) mode=life2; slug=shishya-brain-life2; sources="\"$user/shishya-brain-life\", \"$user/shishya-brain-v2\"" ;;
  *) echo "unknown flag $flag" >&2; exit 2 ;;
esac

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

kdir="$out/kernel-$slug"
rm -rf "$kdir" && mkdir -p "$kdir"
sed "s/^MODE = \"full\"/MODE = \"$mode\"/" "$here/kernel.py" > "$kdir/kernel.py"
printf '{"id": "%s/%s", "title": "%s", "code_file": "kernel.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": ["%s/shishya-brain-code"], "competition_sources": [], "kernel_sources": [%s]}\n' \
  "$user" "$slug" "$slug" "$user" "$sources" > "$kdir/kernel-metadata.json"
"$kaggle" kernels push -p "$kdir"

echo "running on Kaggle..."
for _ in $(seq 1 600); do
  s=$("$kaggle" kernels status "$user/$slug" 2>&1 | tail -1)
  echo "$(date -u +%H:%M) $s"
  case "$s" in *COMPLETE*) break ;; *ERROR*|*CANCEL*) "$kaggle" kernels output "$user/$slug" -p "$out/kaggle-$slug" || true; exit 1 ;; esac
  sleep 60
done
mkdir -p "$out/kaggle-$slug"
"$kaggle" kernels output "$user/$slug" -p "$out/kaggle-$slug"
echo "output: $out/kaggle-$slug/out/"
