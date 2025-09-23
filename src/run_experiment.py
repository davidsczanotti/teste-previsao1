# src/run_experiment.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# imports do projeto
from src.ingest import get_prices
from src.features import make_long_df, add_ta_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rodar experimento NHITS -> Sinais -> Backtest")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["VALE3.SA", "PETR4.SA", "BOVA11.SA", "ITUB4.SA"],
        help="Lista de tickers (yfinance).",
    )
    p.add_argument("--start", type=str, default="2020-01-01", help="Data inicial (YYYY-MM-DD).")
    p.add_argument("--horizon", type=int, default=5, help="Horizonte de previsão (dias).")
    p.add_argument("--n-windows", type=int, default=8, help="Janelas de CV rolling para NHITS.")
    p.add_argument("--input-size", type=int, default=60, help="Janela (lookback) do NHITS.")
    p.add_argument("--max-steps", type=int, default=300, help="Passos de treino por janela.")
    p.add_argument("--seed", type=int, default=1, help="Random seed do modelo.")
    p.add_argument("--fees", type=float, default=0.0005, help="Taxa de corretagem por trade.")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage por trade.")
    p.add_argument("--init-cash", type=float, default=100_000.0, help="Capital inicial do backtest.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # 1) Baixar dados
    print("1) Baixando dados...")
    close_wide: pd.DataFrame = get_prices(args.tickers, start=args.start)
    # print sample igual ao seu estilo
    print(close_wide.head().to_string())

    # 2) Preparar long_df + features
    print("2) Preparando long_df + features básicas...")
    long_df = make_long_df(close_wide)
    long_df = add_ta_features(long_df)

    # 3) Treinar NHITS e prever h
    print(f"3) Treinando NHITS (rolling) e prevendo h={args.horizon}...")
    yhat_df: pd.DataFrame = train_predict_nhits(
        df_long=long_df,  # <<< CORREÇÃO AQUI
        horizon=args.horizon,
        n_windows=args.n_windows,
        input_size=args.input_size,
        max_steps=args.max_steps,
        random_seed=args.seed,
        start_padding_enabled=True,
        verbose=True,
    )
    # Mostro 5 linhas de exemplo (como você costuma ver)
    with pd.option_context("display.max_rows", 10):
        print(yhat_df.head().to_string())

    # 4) Gerar sinais a partir das previsões
    print("4) Gerando sinais a partir das previsões...")
    # ajuste o exp_thresh se quiser mais/menos trades
    signals = build_signals_from_forecast(
        yhat_df=yhat_df,
        close_wide=close_wide,
        forecast_horizon=args.horizon,
        exp_thresh=1.002,
    )

    # 5) Backtest por ticker (vectorbt)
    print("5) Backtest por ticker (vectorbt)...\n")
    summary_df, _ = run_backtest(
        close_wide=close_wide,
        signals=signals,
        init_cash=args.init_cash,
        fees=args.fees,
        slippage=args.slippage,
        direction="longonly",
        save_trades=True,
        report_path="reports/summary_baseline.csv",
    )

    print("\n==== RESUMO ====\n")
    # mesma ordem de colunas do seu relatório
    cols = ["Total Return [%]", "Sharpe Ratio", "Win Rate [%]", "Max Drawdown [%]", "Trades"]
    summary_df = summary_df.reindex(columns=cols)
    print(summary_df.to_string())
    print("\nRelatório salvo em: reports/summary_baseline.csv")


if __name__ == "__main__":
    Path("reports").mkdir(parents=True, exist_ok=True)
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
