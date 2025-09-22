# src/run_experiment.py (trecho principal intacto, só mudam as chamadas)
from __future__ import annotations
import pandas as pd
from pandas.tseries.offsets import BDay
from src.ingest import get_prices
from src.features import make_long_df, add_ta_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtests, summarize_portfolios

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA"]

def main():
    print("1) Baixando dados...")
    end = (pd.Timestamp.today().normalize() - BDay(7)).date().isoformat()
    close = get_prices(TICKERS, start="2016-01-01", end=end)
    print(close.tail())

    print("2) Preparando long_df + features básicas...")
    long_df = make_long_df(close)
    long_df = add_ta_features(long_df)

    print("3) Treinando NHITS (rolling) e prevendo h=5...")
    yhat_df = train_predict_nhits(
        long_df, h=5, input_size=60, max_steps=300, freq='D',
        n_windows=52, step_size=5  # ~5 anos de janelas semanais, ajuste à vontade
    )
    print(yhat_df.tail())

    print("4) Gerando sinais a partir das previsões...")
    signals = build_signals_from_forecast(
        close_wide=close,
        yhat_df=yhat_df,
        horizon=5,
        exp_thresh=0.001,           # 0,1% fixo
        use_vol_threshold=False      # mude para True p/ limiar dinâmico
    )

    print("5) Backtest por ticker (vectorbt)...")
    portfolios = run_backtests(close, signals, fees=0.001, slippage=0.0005, init_cash=100_000)
    summary = summarize_portfolios(portfolios)
    print("\n==== RESUMO ====\n")
    print(summary)

    summary.to_csv("reports/summary_baseline.csv", float_format="%.4f")
    print("\nRelatório salvo em: reports/summary_baseline.csv")

if __name__ == "__main__":
    main()
