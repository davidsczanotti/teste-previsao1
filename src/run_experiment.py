# src/run_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["VALE3.SA", "PETR4.SA", "BOVA11.SA", "ITUB4.SA"],
    )
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--init-cash", type=float, default=100_000.0)
    p.add_argument("--fees", type=float, default=0.0005)
    p.add_argument("--slippage", type=float, default=0.0005)
    # Parâmetros extras do modelo
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--input-size", type=int, default=0)  # 0 => auto
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("1) Baixando dados...")
    close_wide = get_prices(args.tickers, start=args.start)
    print(close_wide.tail(5))

    print("2) Preparando long_df + features básicas...")
    long_df = prepare_long_and_features(close_wide)

    print("3) Treinando NHITS (rolling) e prevendo h=5...")
    yhat_df = train_predict_nhits(
        df_long=long_df[["unique_id", "ds", "y"]],  # apenas o target
        horizon=args.horizon,
        max_steps=args.max_steps,
        n_windows=args.n_windows,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        input_size=(args.input_size if args.input_size and args.input_size > 0 else None),
        # step_size default = horizon dentro da função
        # start_padding_enabled=True está dentro do models_ts
    )

    print(yhat_df.head())

    print("4) Gerando sinais a partir das previsões...")
    signals = build_signals_from_forecast(
        forecast_df=yhat_df,
        close_wide=close_wide,
        # (use os seus parâmetros padrão da função; já aplicamos shift(1) no backtest)
    )

    print("5) Backtest por ticker (vectorbt)...\n")
    summary_df, _ = run_backtest(
        close_wide=close_wide,
        signals=signals,
        init_cash=args.init_cash,
        fees=args.fees,
        slippage=args.slippage,
        direction="longonly",
        report_path=Path("reports/summary_baseline.csv"),
    )

    print("\n==== RESUMO ====\n")
    print(summary_df)
    print("\nRelatório salvo em: reports/summary_baseline.csv")


if __name__ == "__main__":
    main()
