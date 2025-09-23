from __future__ import annotations

import argparse

from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models import train_nhits_rolling, forecast_last_window
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest


def main() -> None:
    # ------------------------------------------------------------------ #
    # CLI: custos e caixa inicial
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(description="Rodar experimento NHITS + geração de sinais + backtest")
    parser.add_argument(
        "--fees",
        type=float,
        default=0.0005,
        help="Taxa por operação (fração). Ex: 0.0005 = 5 bps",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage por operação (fração). Ex: 0.0005 = 5 bps",
    )
    parser.add_argument(
        "--init-cash",
        type=float,
        default=100_000.0,
        help="Caixa inicial do portfólio",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Horizonte de previsão (nº de candles à frente)",
    )
    parser.add_argument(
        "--exp-thresh",
        type=float,
        default=0.003,
        help="Limiar de retorno esperado para gerar entrada/saída (ex.: 0.003 = 0,3%)",
    )
    args = parser.parse_args()

    # 1) Dados
    print("1) Baixando dados...")
    close = get_prices()  # DataFrame wide: colunas = tickers, index = datas

    # 2) Long + features
    print("2) Preparando long_df + features básicas...")
    long_df = prepare_long_and_features(close)

    # 3) Treino + previsão
    print(f"3) Treinando NHITS (rolling) e prevendo h={args.horizon}...")
    model = train_nhits_rolling(long_df)
    yhat_df = forecast_last_window(model, horizon=args.horizon)

    # 4) Sinais
    print("4) Gerando sinais a partir das previsões...")
    signals = build_signals_from_forecast(
        close_wide=close,
        yhat_df=yhat_df,
        horizon=args.horizon,
        exp_thresh=args.exp_thresh,
    )

    # 5) Backtest
    print("5) Backtest por ticker (vectorbt)...")
    summary_df, _ = run_backtest(
        close_wide=close,
        signals=signals,
        init_cash=args.init_cash,
        fees=args.fees,
        slippage=args.slippage,
        report_path="reports/summary_baseline.csv",
    )

    # saída amigável
    print("\n==== RESUMO ====\n")
    print(summary_df.to_string())
    print("\nRelatório salvo em: reports/summary_baseline.csv")


if __name__ == "__main__":
    main()
