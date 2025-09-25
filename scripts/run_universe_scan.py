from __future__ import annotations

"""
scripts/run_universe_scan.py

Varredura do universo de tickers em batches, usando um perfil leve de scan
baseado no final_A e salvando ranking por métricas no CSV e no registry.

Uso:
    poetry run python -m scripts.run_universe_scan \
        --universe configs/universe_b3.txt \
        --batch-size 10 \
        --start 2018-01-01 \
        --lead 2 --trend-sma 100 --dyn-k 0.8 --exp 0.0 \
        --atr-window 14 --atr-k 1.5 \
        --n-windows 12 --step-size 10 --max-steps 30
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest
from src.config import Cfg
from src.exp_store import log_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universe scan and ranking")
    p.add_argument("--universe", type=str, default="configs/universe_b3.txt")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--start", type=str, default="2018-01-01")
    # modelo
    p.add_argument("--n-windows", type=int, default=12)
    p.add_argument("--step-size", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--lead", type=int, default=2)
    # sinais
    p.add_argument("--trend-sma", type=int, default=100)
    p.add_argument("--dyn-k", type=float, default=0.8)
    p.add_argument("--exp", type=float, default=0.0)
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-k", type=float, default=1.5)
    # backtest
    p.add_argument("--risk", type=float, default=None)
    # ranking/quality filters
    p.add_argument("--min-trades", type=int, default=5, help="Min. trades per ticker to keep in ranking")
    return p.parse_args()


def read_universe(path: str) -> List[str]:
    tickers: List[str] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)
    return tickers


def scan_batch(tickers: List[str], args: argparse.Namespace) -> pd.DataFrame:
    close = get_prices(tickers=tickers, start=args.start)
    close = close.sort_index()

    prep_out = prepare_long_and_features(close)
    long_df = prep_out if isinstance(prep_out, pd.DataFrame) else prep_out["long_df"]

    yhat = train_predict_nhits(
        long_df=long_df,
        horizon=5,
        input_size=60,
        n_windows=args.n_windows,
        step_size=args.step_size,
        max_steps=args.max_steps,
        seed=1,
        lead_for_signal=args.lead,
    )

    signals = build_signals_from_forecast(
        forecast_df=yhat,
        close_wide=close,
        exp_thresh=args.exp,
        consec=1,
        trend_sma=args.trend_sma,
        dyn_thresh_k=args.dyn_k,
        vol_window=20,
        rsi_window=None,
        rsi_min=None,
        bb_window=None,
        bb_k=2.0,
        atr_window=args.atr_window,
        atr_stop_k=args.atr_k,
        cooldown_bars=0,
        max_hold_bars=None,
        only_non_overlapping=True,
        debug=False,
    )

    size_wide = None
    if args.risk:
        vol = close.pct_change().rolling(20, min_periods=20).std().reindex(close.index).fillna(method="ffill").replace(0.0, 1e-4)
        risk_cash = args.risk * 100_000
        size_wide = (risk_cash / (vol * close)).replace([float('inf'), float('-inf')], float('nan')).fillna(0.0).clip(lower=0.0)
        size_wide = size_wide.applymap(lambda x: float(max(0, int(x))))

    summary, pf_dict = run_backtest(
        close_wide=close,
        signals=signals,
        init_cash=100_000.0,
        fees=0.0005,
        slippage=0.0005,
        direction="longonly",
        save_trades=False,
        report_path="reports/summary_scan_batch.csv",
        size_wide=size_wide,
        aggregate_portfolio=True,
    )

    # log simplified run
    cfg = Cfg.parse_obj({
        "data": {"tickers": tickers, "start": args.start},
        "model": {"horizon": 5, "input_size": 60, "n_windows": args.n_windows, "step_size": args.step_size, "max_steps": args.max_steps, "seed": 1, "lead_for_signal": args.lead},
        "signals": {"exp_thresh": args.exp, "consec": 1, "trend_sma": args.trend_sma, "dyn_thresh_k": args.dyn_k, "vol_window": 20, "rsi_window": None, "rsi_min": None, "bb_window": None, "bb_k": 2.0, "atr_window": args.atr_window, "atr_stop_k": args.atr_k, "cooldown_bars": 0, "max_hold_bars": None},
        "backtest": {"init_cash": 100000, "fees": 0.0005, "slippage": 0.0005, "direction": "longonly", "only_non_overlapping": True, "risk_per_trade": args.risk},
        "tracking": {"use_mlflow": False, "mlflow_experiment": "scan", "mlflow_uri": None},
        "registry": {"enabled": True, "path": "reports/experiments.sqlite"},
        "experiment": {"name": "universe_scan", "notes": "scan leve"},
    })
    try:
        log_run("reports/experiments.sqlite", cfg=cfg, summary_df=summary, report_path="reports/summary_scan_batch.csv")
    except Exception:
        pass

    return summary.reset_index()


def main() -> None:
    args = parse_args()
    universe = read_universe(args.universe)
    if not universe:
        raise SystemError("Universo vazio")

    Path("reports").mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(0, len(universe), args.batch_size):
        batch = universe[i : i + args.batch_size]
        print(f"[scan] Batch {i//args.batch_size+1}: {batch}")
        try:
            summary_batch = scan_batch(batch, args)
            rows.append(summary_batch)
        except Exception as exc:
            print(f"[scan] Falha no batch {batch}: {exc}")

    if rows:
        df = pd.concat(rows, ignore_index=True)
        # Filtro por número mínimo de trades
        df = df.copy()
        if "Trades" in df.columns:
            df = df[df["Trades"].fillna(0).astype(int) >= args.min_trades]

        # Ranking robusto a NaNs: empurra NaN para o fim
        sharpe = pd.to_numeric(df.get("Sharpe Ratio"), errors="coerce").fillna(-1e12)
        ret = pd.to_numeric(df.get("Total Return [%]"), errors="coerce").fillna(-1e12)
        df["Sharpe Rank"] = sharpe.rank(ascending=False, method="min")
        df["Return Rank"] = ret.rank(ascending=False, method="min")
        df["Score"] = 0.6 * (1.0 / df["Sharpe Rank"]) + 0.4 * (1.0 / df["Return Rank"])
        df = df.sort_values(["Score"], ascending=False)

        # Saídas: ranking principal e cópia datada
        out = Path("reports/universe_rankings.csv")
        df.to_csv(out, index=False)
        print(f"[scan] Ranking salvo em {out}")

        from datetime import datetime, timezone
        day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        daily_dir = Path(f"reports/daily/{day}")
        daily_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(daily_dir / "universe_rankings.csv", index=False)
        print(f"[scan] Cópia diária salva em {daily_dir / 'universe_rankings.csv'}")
    else:
        print("[scan] Nada para salvar")


if __name__ == "__main__":
    main()
