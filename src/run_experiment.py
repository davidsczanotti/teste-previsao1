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

# Escolha por ativo (com base nos resultados anteriores):
horizons = {"VALE3.SA": 5, "ITUB4.SA": 5, "PETR4.SA": 2, "BOVA11.SA": 3}  # ↓ de 3 para 2 (única mudança deste passo)

volks = {
    "VALE3.SA": 0.35,
    "ITUB4.SA": 0.25,
    "PETR4.SA": 0.50,  # ↑ de 0.45 -> 0.50 (mais seletivo)
    "BOVA11.SA": 0.25
}

min_holds = {
    "ITUB4.SA": 1,  # mantém
    "VALE3.SA": 2,  # mantém
    "PETR4.SA": 1,  # << de 2 para 1 (permite sair no D+1 se virar)
    "BOVA11.SA": 2,  # mantém
}


def main():
    print("1) Baixando dados...")
    end = (pd.Timestamp.today().normalize() - BDay(7)).date().isoformat()
    close = get_prices(TICKERS, start="2016-01-01", end=end)
    print(close.tail())

    print("2) Preparando long_df + features básicas...")
    long_df = make_long_df(close)
    long_df = add_ta_features(long_df)

    print("3) Treinando NHITS (rolling) e prevendo h=5...")
    yhat_df = train_predict_nhits(long_df, h=3, input_size=60, max_steps=300, freq="D", n_windows=260, step_size=1)
    print(yhat_df.tail())

    print("4) Gerando sinais a partir das previsões...")
    signals = build_signals_from_forecast(
        close_wide=close,
        yhat_df=yhat_df,
        horizon=horizons,  # {VALE3:5, ITUB4:5, PETR4:2, BOVA11:3}
        use_vol_threshold=True,
        vol_k=volks,  # {VALE3:0.35, ITUB4:0.25, PETR4:0.45, BOVA11:0.25}
        early_exit_on_flip=True,
        min_hold=min_holds,  # {ITUB4:1, VALE3:2, PETR4:1, BOVA11:2}
        exit_symmetric=False,
        trend_sma={"PETR4.SA": 200},  # <<< ÚNICA NOVIDADE DESTE PASSO
    )

    # DEBUG: quantos sinais por ticker?
    for t in TICKERS:
        ent = signals[t]["entries"].sum()
        exi = signals[t]["exits"].sum()
        print(f"DEBUG {t}: entries={int(ent)} exits={int(exi)}")

    print("5) Backtest por ticker (vectorbt)...")
    portfolios = run_backtests(close, signals, fees=0.001, slippage=0.0005, init_cash=100_000)
    summary = summarize_portfolios(portfolios)
    print("\n==== RESUMO ====\n")
    print(summary)

    summary.to_csv("reports/summary_baseline.csv", float_format="%.4f")
    print("\nRelatório salvo em: reports/summary_baseline.csv")


if __name__ == "__main__":
    main()
