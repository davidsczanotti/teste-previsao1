#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."
mkdir -p reports/logs
dt="$(date +%Y%m%d_%H%M%S)"

poetry run python -m src.run_experiment --config configs/forward_test.yaml | tee "reports/logs/nightly_forward_${dt}.log"

if [ -f reports/portfolio_combined.csv ]; then
  cp -f reports/portfolio_combined.csv "reports/forward_portfolio_combined_${dt}.csv"
  poetry run python - << 'PY'
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
p = Path('reports/portfolio_combined.csv')
df = pd.read_csv(p, parse_dates=['date'])
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['equity'], label='Portfolio Combined (Forward)')
plt.xlabel('Date'); plt.ylabel('Equity'); plt.title('Curva de capital – Forward Test'); plt.legend(); plt.tight_layout()
out = Path('reports')/('forward_equity_' + __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S') + '.png')
plt.savefig(out, dpi=150)
print(f'Saved {out}')
PY
fi

echo "[nightly_forward_test] Done at ${dt}"

