# AGENTS.md

> **Purpose**: Give code-generation agents (Codex/Copilot/etc.) a concise but rigorous briefing so they can make **correct** edits to this repo without breaking core assumptions (no-lookahead, reproducibility, stable interfaces, and cross-version compatibility with `vectorbt` and `neuralforecast`).

---

## 0) TL;DR for Agents

- **Do not create look-ahead bias**. Signals must be **executed on the next bar**: `entries = entries.shift(1, fill_value=False)` and same for `exits` (handled in `src/backtest.py`).
- **Always align Series** to the **price index** before backtesting (`reindex(..., fill_value=False)`). Never call `.values` to align.
- **Forecast format** is the NixTLA/NeuralForecast standard: `unique_id`, `ds`, `y_hat`.
- **Signals contract**: `Dict[str, Dict[str, pd.Series]]` -> `{ticker: {"entries": bool Series, "exits": bool Series}}` **aligned to the same DateTimeIndex as prices**.
- **Backtest** uses `vectorbt.Portfolio.from_signals(accumulate=True)`; we log basic metrics and save `reports/summary_baseline.csv`.
- **Cross-version safety**: Use helper accessors in backtest to compute Sharpe/MaxDD/WinRate, because `vectorbt` column names vary across versions.
- If you add model params, **expose them in `src/run_experiment.py` CLI** and **document defaults**.
- If you add a new strategy, put it under `src/signals.py` (pure functions), and keep **I/O contracts** stable.

---

## 1) Repository Topology (canonical)

```
.
├── pyproject.toml              # poetry config + pkg deps
├── README.md                   # project overview / how to run
├── AGENTS.md                   # this file
├── data/                       # (optional) local caches
├── reports/                    # outputs: summary + per-ticker trades
├── notebooks/
│   └── quick_report.ipynb      # visual sanity-check of summary
├── scripts/
│   └── test_backtest_smoke.py  # shift(1) + plumbing smoke test
└── src/
    ├── __init__.py
    ├── ingest.py               # get_prices(tickers, start, end) -> close_wide
    ├── prep.py                 # prepare_long_and_features(close_wide) -> long_df (+basic feats)
    ├── models_ts.py            # NHITS training/prediction helpers
    ├── signals.py              # rule(s) to turn y_hat into entries/exits
    ├── backtest.py             # run_backtest(...) cross-version-safe metrics
    └── run_experiment.py       # end-to-end CLI
```

> **Note**: Some earlier issues came from mismatched names/paths (`src.prep` missing, etc.). Keep modules under `src/` and import with `from src.module import ...`.

---

## 2) Canonical Data Contracts

### 2.1 `ingest.get_prices`

- **Input**: `tickers: List[str]`, `start: str|pd.Timestamp`, `end: Optional[str]`.
- **Output**: `close_wide: pd.DataFrame`
  - Columns: tickers
  - Index: **DatetimeIndex (trading days)**, **monotonic increasing**, **unique**
  - Values: close prices (float).

**Agent rules**:
- If you add vendor adapters (e.g., yfinance/polars), keep the **same shape**.
- **Never** return duplicate timestamps. If vendor returns dupes, call `.groupby(level=0).last()` or `.asfreq("B")` cautiously.

### 2.2 `prep.prepare_long_and_features`

- **Input**: `close_wide`
- **Output**: `long_df` with columns: `unique_id` (ticker), `ds` (timestamp), `y` (target), and **optional features** (e.g., rolling means).

**Agent rules**:
- No missing `ds`/`y` within each `unique_id` if avoidable.
- If you add features, prefix them consistently (e.g., `feat_`), and **document** in the docstring.

### 2.3 `models_ts.train_predict_nhits`

**Signature (current)**:
```python
def train_predict_nhits(
    long_df: "pd.DataFrame",
    horizon: int = 5,
    max_steps: int = 300,
    n_windows: int = 8,
    step_size: int | None = None,
    input_size: int = 60,
    start_padding_enabled: bool = True,
    seed: int = 1,
) -> "pd.DataFrame":  # columns: unique_id, ds, y_hat
    ...
```

**Behavior**:
- Wraps `neuralforecast` NHITS in rolling windows (via `NeuralForecast.cross_validation`).
- Returns **stacked** forecast DataFrame with **exactly**: `unique_id`, `ds`, `y_hat`.

**Agent rules**:
- Don’t pass unknown kwargs (e.g., `use_gpu`, `verbose`, `random_seed`) unless you also update the function signature.
- Handle **short series**: either reduce `input_size` or set `start_padding_enabled=True`.
- Keep `seed` default stable for reproducibility.

### 2.4 `signals.build_signals_from_forecast`

**Signature (core idea)**:
```python
def build_signals_from_forecast(
    yhat_df: pd.DataFrame,
    close_wide: pd.DataFrame,
    threshold: float = 0.0,
    only_non_overlapping: bool = True,
    trend_sma: int | dict[str, int] | None = None,
) -> dict[str, dict[str, pd.Series]]:
    ...
```
**Contract**:
- For each `ticker`, returns **aligned** boolean Series:
  - `entries`: True means **enter** long **on next bar** (shift happens in backtest).
  - `exits`: True means **exit** position **on next bar**.
- Must **reindex to `close_wide.index`** before returning (fill gaps with `False`).

**Agent rules**:
- Optionally filter entries by trend (e.g., `close > SMA(trend_window)`).
- `only_non_overlapping=True` should suppress overlapping entries while position is open.

### 2.5 `backtest.run_backtest`

- **Input**: `close_wide`, `signals`, `init_cash`, `fees`, `slippage`, `direction="longonly"`, `report_path="reports/summary_baseline.csv"`.
- **Mechanics**:
  - Aligns signals to price index and **applies `shift(1)`** (**no-lookahead invariant**).
  - Calls `vectorbt.Portfolio.from_signals(..., accumulate=True)`.
  - Computes cross-version-safe metrics:
    - `Total Return [%]`
    - `Sharpe Ratio`
    - `Win Rate [%]`
    - `Max Drawdown [%]` **as positive magnitude**
    - `Trades`
  - Saves per-ticker trades under `reports/trades_{ticker}.csv` when available.
  - Saves summary CSV to `report_path`.

**Agent rules**:
- If you add metrics, **don’t break the summary schema**; append columns at the end.
- `frequency` may be absent; Sharpe warnings are OK. Prefer safe accessors (`_safe_*` helpers).

---

## 3) End-to-End CLI (`src/run_experiment.py`)

**Canonical usage**:
```bash
poetry run python -m src.run_experiment \
  --tickers VALE3.SA PETR4.SA BOVA11.SA ITUB4.SA \
  --start 2020-01-01 \
  --horizon 5 \
  --max-steps 300 \
  --n-windows 8 \
  --input-size 60 \
  --fees 0.0005 --slippage 0.0005 --init-cash 100000
```

**What it does**:
1) Ingest prices → `close_wide` (unique, sorted index).  
2) Prepare long format + basic features → `long_df`.  
3) Train/predict NHITS via rolling CV → `yhat_df`.  
4) Build rule-based signals from forecast → `signals`.  
5) Run vectorbt backtest with **shift(1)** → CSV report in `reports/`.

**Agent rules**:
- Keep printed headers stable (used by earlier logs/tests).  
- Validate flag combos early; give actionable errors (e.g., horizon > input_size).

---

## 4) Guardrails & Gotchas (Lessons from fixes)

1) **Duplicate index / reindex errors**  
   - Always ensure `close_wide.index` is **unique** and monotonic. If vendor gives duplicates: `close_wide = close_wide[~close_wide.index.duplicated(keep='last')]`.
   - When you reindex forecast series: `s.reindex(close.index, fill_value=False)`.

2) **No-lookahead**  
   - Enforced in `backtest.run_backtest`: `entries/exits = entries/exits.shift(1, fill_value=False)`.
   - Do not shift twice (don’t shift inside `signals.py`).

3) **Vectorbt cross-version fields**  
   - Use `_safe_total_return`, `_safe_sharpe`, `_safe_max_dd`, `_win_rate_and_trades`.  
   - `PnL` vs `Pnl`, `Entry Time` vs fields in `records` vary.

4) **NeuralForecast short series**  
   - Error: *“Time series is too short for training”*. Fix by smaller `input_size` and/or `start_padding_enabled=True`.  
   - CI-friendly defaults: `input_size=60`, `n_windows` moderate.

5) **Unknown kwargs**  
   - Passing `use_gpu`, `verbose`, `random_seed` to old signatures caused `TypeError`.  
   - Only pass arguments **declared** in the current function signature.

6) **Sharpe warnings**  
   - If `frequency` isn’t set, vectorbt warns. We accept warnings and compute Sharpe via safe accessor (may be NaN).

7) **Module path mismatches**  
   - Keep imports as `from src.module import ...`. Ensure files exist and `__init__.py` is present.

---

## 5) Extensibility Recipes

### 5.1 Add a new forecast model
- Create `src/models_mynew.py`: expose `train_predict_mynew(long_df, ...) -> yhat_df` in **same schema** (`unique_id, ds, y_hat`).
- Wire into `run_experiment.py` behind a `--model` flag; default remains `nhits`.

### 5.2 Add a new strategy
- Implement a pure function in `src/signals.py`: `build_signals_from_forecast_X(...)` that returns the **signals contract**.  
- Add a CLI flag `--strategy` and route accordingly.

### 5.3 Add features
- Extend `prep.prepare_long_and_features`.  
- **Document** new feature names and their roll windows. Keep deterministic seeds where relevant.

---

## 6) Testing & Diagnostics

### 6.1 Smoke test
Run:
```bash
poetry run python -m scripts.test_backtest_smoke
```
Asserts:
- At least one trade.
- `entries/exits` are shifted by 1 bar (no same-bar execution).
- Summary table has expected columns.

### 6.2 Quick sanity check notebook
Open `notebooks/quick_report.ipynb` to see bar charts of `Total Return [%]` by ticker from the latest `reports/summary_baseline.csv`.

### 6.3 Common failure messages
- `ModuleNotFoundError: No module named 'src.prep'` → file missing or import path wrong.
- `ValueError: cannot reindex on an axis with duplicate labels` → de-duplicate `close_wide.index`.
- `TypeError: train_predict_nhits() got an unexpected keyword argument 'X'` → signature drift; remove unknown kwarg or update signature.
- `Time series is too short for training` → lower `input_size` or enable start padding.

---

## 7) Coding Standards for Agents

- **Type hints** everywhere. Keep functions small and side-effect-free.
- **Docstrings** must describe **I/O contracts** and failure modes.
- **No silent except**: catch `Exception` only to **re-raise with context** or log + use safe fallbacks (as in backtest metrics).
- **No hard-coded tickers** in library code; only in top-level scripts or tests.
- **Determinism**: keep `seed=1` default in modeling functions.
- **Filesystem**: create directories (`Path(...).mkdir(parents=True, exist_ok=True)`) before writing.
- **Logging**: minimal `print()` with clear prefixes. Avoid noisy repeated logs.

---

## 8) API Cheat Sheet

### `src.models_ts.train_predict_nhits`
```python
yhat_df = train_predict_nhits(
    long_df=long_df,
    horizon=5,
    max_steps=300,
    n_windows=8,
    step_size=None,           # auto default by library
    input_size=60,
    start_padding_enabled=True,
    seed=1,
)
# -> columns: unique_id, ds, y_hat
```

### `src.signals.build_signals_from_forecast`
```python
signals = build_signals_from_forecast(
    yhat_df=yhat_df,
    close_wide=close_wide,
    threshold=0.0,
    only_non_overlapping=True,
    trend_sma= None or 200 or {"VALE3.SA": 150, "BOVA11.SA": 200},
)
# -> {"VALE3.SA": {"entries": Series[bool], "exits": Series[bool]}, ...}
```

### `src.backtest.run_backtest`
```python
summary_df, portfolios = run_backtest(
    close_wide=close_wide,
    signals=signals,
    init_cash=100_000.0,
    fees=0.0005,
    slippage=0.0005,
    direction="longonly",
    report_path="reports/summary_baseline.csv",
)
```

---

## 9) Roadmap Hints for Agents

- Add slippage/fees as CLI flags (already supported) and propagate to backtest.
- Add **position sizing** (currently 100% when entering; could extend to fractional sizing).
- Extend strategies (e.g., take-profit/stop-loss based on forecast error bands).
- Plug a feature store or macro signals (keep contracts stable).

---

## 10) Attribution

This brief encodes all constraints and conventions we established while iterating live:
- **No-lookahead**, **safe reindex**, **cross-version metric access**, **NHITS rolling CV**, and **CLI-driven experiments**.

Agents should adhere to these invariants to keep results reproducible and comparable.
