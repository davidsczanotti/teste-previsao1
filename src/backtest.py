# src/backtest.py
import numpy as np
import vectorbt as vbt

def run_backtest(prices, entries, exits, fees=0.001, slippage=0.0005):
    pf = vbt.Portfolio.from_signals(
        close=prices, entries=entries, exits=exits,
        fees=fees, slippage=slippage,
        init_cash=100_000, freq='D'
    )
    return pf
