#!/bin/bash
# One job at a time, at low priority, so nothing else on the machine is slowed.
cd /home/ubuntu/shishya-lab || exit 1
export HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=lab
PY="nice -n 19 /home/ubuntu/shishya/.venv/bin/python"
while pgrep -f '[r]ead_fakes2' >/dev/null; do sleep 20; done
$PY lab/knowing.py test --out lab/out/knowing.json > lab/out/knowing.log 2>&1
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('lab/out/knowing.json'))['layer']['passed'] else 1)"; then
  $PY lab/knowing.py practice --out lab/out/knowing_practice.pt >> lab/out/knowing.log 2>&1
  $PY lab/knowing.py test --layer lab/out/knowing_practice.pt --name practice --out lab/out/knowing.json >> lab/out/knowing.log 2>&1
fi
echo "KNOWING STAGE FINISHED" >> lab/out/knowing.log
$PY lab/life.py --out lab/out/life > lab/out/life.log 2>&1
echo "CHAIN FINISHED" >> lab/out/life.log
