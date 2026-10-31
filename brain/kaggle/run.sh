#!/usr/bin/env bash
# Run the full life test on Kaggle's free GPU with your own account.
#
#   export KAGGLE_API_TOKEN=...            # or ~/.kaggle/kaggle.json
#   brain/kaggle/run.sh YOUR_KAGGLE_USERNAME [branch]
#
# The kernel clones the branch from GitHub (push it first), trains GPT, BDH and the Brain on the same 100M bytes,
# runs the life test, and the results are downloaded to brain/out/kaggle/.
set -euo pipefail
user=${1:?usage: run.sh KAGGLE_USERNAME [branch]}
branch=${2:-$(git rev-parse --abbrev-ref HEAD)}
here="$(cd "$(dirname "$0")" && pwd)"
kaggle="${KAGGLE:-kaggle}"
slug="shishya-brain-life"
kdir="$here/../out/kernel"
mkdir -p "$kdir"
sed "s|BRANCH_NAME|$branch|" "$here/kernel.py" > "$kdir/kernel.py"
printf '{"id": "%s/%s", "title": "%s", "code_file": "kernel.py", "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}\n' \
  "$user" "$slug" "$slug" > "$kdir/kernel-metadata.json"
"$kaggle" kernels push -p "$kdir"

echo "running on Kaggle (a few hours)..."
for _ in $(seq 1 480); do
  s=$("$kaggle" kernels status "$user/$slug" 2>&1 | tail -1)
  case "$s" in *COMPLETE*) break ;; *ERROR*|*CANCEL*) echo "$s" >&2; exit 1 ;; esac
  sleep 60
done
dest="$here/../out/kaggle"
mkdir -p "$dest"
"$kaggle" kernels output "$user/$slug" -p "$dest"
echo "results: $dest/out/life.json"
