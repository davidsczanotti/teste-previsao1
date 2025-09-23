# src/run_experiment.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# IMPORTS RELATIVOS (funcionam com `python -m src.run_experiment`)
from .ingest import get_prices
from .features import make_long_df, add_ta_features
from .models_ts import train_predict_nhits
from .signals import build_signals_from_forecast
from .backtest import run_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline: dados -> features -> NHITS -> sinais -> backtest"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["VALE3.SA", "PETR4.SA", "BOVA11.SA", "ITUB4.SA"],
        help="Lista de tickers (Yahoo/BR).",
    )
    parser.add_argument(
        "--start", type=str, default=None, help="Data inicial (YYYY-MM-DD). Opcional."
    )
    parser.add_argument(
        "--horizon", type=int, default=5, help="Horizonte de previsão (h)."
    )
    parser.add_argument(
        "--max-steps", type=int, default=300, help="Passos de treino (max_steps)."
    )
    parser.add_argument(
        "--fees", type=float, default=0.0005, help="Taxa por trade (proporção)."
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0005, help="Slippage por trade (proporção)."
    )
    parser.add_argument(
        "--init-cash", type=float, default=100_000.0, help="Capital inicial do backtest."
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/summary_baseline.csv",
        help="Caminho do CSV de resumo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Dados
    print("1) Baixando dados...")
    close_wide: pd.DataFrame = get_prices(tickers=args.tickers, start=args.start)
    # garante ordenação de colunas por estética/consistência
    close_wide = close_wide[sorted(close_wide.columns)]
    print(close_wide.tail(5))

    # 2) Long DF + features
    print("2) Preparando long_df + features básicas...")
    long_df = make_long_df(close_wide)
    long_df = add_ta_features(long_df)

    # 3) Treino/Previsão NHITS
    print(f"3) Treinando NHITS (rolling) e prevendo h={args.horizon}...")
    yhat_df = train_predict_nhits(
        long_df=long_df,
        h=args.horizon,
        max_steps=args.max_steps,
        n_windows=None,         # deixar None para usar janela padrão (rolling)
        random_seed=1,
        use_gpu=False,
        verbose=True,
    )
    # Mostra só as últimas linhas de um ticker como você costuma ver
    # (normalmente VALE3.SA aparece no log)
    print(yhat_df.tail(5))

    # 4) Sinais a partir das previsões
    print("4) Gerando sinais a partir das previsões...")
    signals = build_signals_from_forecast(
        close_wide=close_wide,
        yhat_df=yhat_df,
        horizon=args.horizon,
        exp_thresh=0.003,  # mesmo padrão que você já vinha usando
    )

    # 5) Backtest
    print("5) Backtest por ticker (vectorbt)...\n")
    summary_df, _ = run_backtest(
        close_wide=close_wide,
        signals=signals,
        init_cash=args.init_cash,
        fees=args.fees,
        slippage=args.slippage,
        direction="longonly",
        save_trades=True,
        report_path=args.report_path,
    )

    print("\n==== RESUMO ====\n")
    if not summary_df.empty:
        print(summary_df)
    else:
        print("Sem resultados (verifique os sinais).")

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.report_path)
    print(f"\nRelatório salvo em: {args.report_path}")


if __name__ == "__main__":
    main()
