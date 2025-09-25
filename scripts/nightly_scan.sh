#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
mkdir -p reports/logs
dt="$(date +%Y%m%d_%H%M%S)"

poetry run python -m scripts.run_universe_scan \
  --universe configs/universe_b3.txt \
  --batch-size 30 \
  --start 2018-01-01 \
  --lead 2 --trend-sma 100 --dyn-k 0.8 --exp 0.0 \
  --atr-window 14 --atr-k 1.5 \
  --n-windows 12 --step-size 10 --max-steps 30 \
  | tee "reports/logs/nightly_scan_${dt}.log"

echo "[nightly_scan] Done at ${dt}"

