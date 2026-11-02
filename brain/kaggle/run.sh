#!/usr/bin/env bash
# Run the life test on Kaggle's free GPU with your own account.
#
#   export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
#   brain/kaggle/run.sh YOUR_KAGGLE_USERNAME [--canary | --brain2 | --life2 | --v3 | --canary3 | --v4 | --canary4]
#
# (no flag)  GPT, BDH and the v1 Brain on the same 100M bytes, then the life test   kernel shishya-brain-life
# --canary   the same, 20 steps each and a short life: GPU-only bugs in minutes     kernel shishya-brain-canary
# --brain2   train the v2 Brain only                                                 kernel shishya-brain-v2
# --life2    life test of v2 against the first run's GPT and BDH (needs both done)  kernel shishya-brain-life2
# --v3       the v3 Brain (v2's network, v3's memory), life test vs the same GPT, BDH  kernel shishya-brain-v3
# --canary3  the same with a short life: GPU-only bugs in minutes                      kernel shishya-brain-canary3
# --v4, --canary4  the same for the v4 Brain                                          kernel shishya-brain-v4 / -canary4
# --v5, --canary5  v5 at scale: SmolLM2-1.7B + v5's memory (docs/V5.md)              kernel shishya-brain-v5 / -canary5
# --v5brain, --canary5brain  v5 with no LLM: the v2 BDH Brain + v5's memory          kernel shishya-brain-v5brain
# --v6brain, --canary6brain  v6 (docs/V6.md) on the Brain                              kernel shishya-brain-v6brain
#
# Uploads brain/ (code only) as a PRIVATE dataset <user>/shishya-brain-code, pushes a private GPU kernel, waits,
# and downloads its output to brain/out/kaggle-<kernel>/.
set -euo pipefail
user=${1:?usage: run.sh KAGGLE_USERNAME [--canary | --brain2 | --life2 | --v3 | --canary3 | --v4 | --canary4 | --v5 | --canary5 | --v5brain | --canary5brain | --v6brain | --canary6brain]}
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
  --v3) mode=v3; slug=shishya-brain-v3; sources="\"$user/shishya-brain-life2\"" ;;
  --canary3) mode=canary3; slug=shishya-brain-canary3; sources="\"$user/shishya-brain-life2\"" ;;
  --v4) mode=v4; slug=shishya-brain-v4; sources="\"$user/shishya-brain-life2\"" ;;
  --canary4) mode=canary4; slug=shishya-brain-canary4; sources="\"$user/shishya-brain-life2\"" ;;
  --v5) mode=v5; slug=shishya-brain-v5 ;;
  --canary5) mode=canary5; slug=shishya-brain-canary5 ;;
  --v5brain) mode=v5brain; slug=shishya-brain-v5brain; sources="\"$user/shishya-brain-life2\"" ;;
  --canary5brain) mode=canary5brain; slug=shishya-brain-canary5brain; sources="\"$user/shishya-brain-life2\"" ;;
  --v6brain) mode=v6brain; slug=shishya-brain-v6brain; sources="\"$user/shishya-brain-life2\", \"$user/shishya-brain-v5brain\"" ;;
  --canary6brain) mode=canary6brain; slug=shishya-brain-canary6brain; sources="\"$user/shishya-brain-life2\", \"$user/shishya-brain-v5brain\"" ;;
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
sleep 60                                   # "ready" can still mean the previous version for a while
for _ in $(seq 1 60); do
  [ "$("$kaggle" datasets status "$user/shishya-brain-code" 2>/dev/null | tail -1)" = "ready" ] && break
  sleep 10
done

kdir="$out/kernel-$slug"
rm -rf "$kdir" && mkdir -p "$kdir"
sed -e "s/^MODE = \"full\"/MODE = \"$mode\"/" -e "s/^COMMIT = \"\"/COMMIT = \"$(cat "$ds/commit.txt")\"/" \
  "$here/kernel.py" > "$kdir/kernel.py"
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
