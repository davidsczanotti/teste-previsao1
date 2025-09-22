# src/backtest.py
from __future__ import annotations
import numpy as np
import pandas as pd
import vectorbt as vbt


def run_backtests(
    close_wide: pd.DataFrame,
    signals: dict,
    fees: float = 0.001,  # 0.1% por trade
    slippage: float = 0.0005,  # 5 bps
    init_cash: float = 100_000,
):
    """
    Roda um backtest por ticker usando vectorbt e retorna dict de Portfolios.
    """
    portfolios = {}
    for ticker, sig in signals.items():
        px = close_wide[ticker].dropna()
        entries = sig["entries"].reindex(px.index).fillna(False)
        exits = sig["exits"].reindex(px.index).fillna(False)

        pf = vbt.Portfolio.from_signals(
            close=close_series,
            entries=entries_series,
            exits=exits_series,
            init_cash=100_000,
            fees=0.001,  # 0,10% por trade (corretagem/IOF/implicit costs)
            slippage=0.001,  # 0,10% de slippage
            direction="longonly",
            accumulate=True,
        )
        trades = pf.trades.records_readable
        trades.to_csv(f"reports/trades_{ticker}.csv", index=False)
        portfolios[ticker] = pf
    return portfolios


def summarize_portfolios(portfolios: dict) -> pd.DataFrame:
    rows = []
    for ticker, pf in portfolios.items():
        stats = pf.stats()
        rows.append(
            {
                "ticker": ticker,
                "Total Return [%]": stats.get("Total Return [%]"),
                "Sharpe Ratio": stats.get("Sharpe Ratio"),
                "Win Rate [%]": stats.get("Win Rate [%]"),
                "Max Drawdown [%]": stats.get("Max Drawdown [%]"),
                "Trades": stats.get("Total Trades"),
            }
        )
    return pd.DataFrame(rows).set_index("ticker").sort_values("Total Return [%]", ascending=False)
