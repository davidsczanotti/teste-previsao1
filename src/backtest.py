# src/backtest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src.metrics import sharpe

try:
    import vectorbt as vbt
except Exception as e:
    raise RuntimeError(
        "vectorbt não instalado/compatível. Verifique a versão (ex.: 0.28.1)."
    ) from e


def _ensure_bool_series(s: Optional[pd.Series], index: pd.Index) -> pd.Series:
    if s is None:
        return pd.Series(False, index=index)
    out = s.reindex(index, fill_value=False)
    if out.dtype != bool:
        out = out.astype(bool)
    return out


def _safe_total_return(pf) -> float:
    try:
        return float(pf.total_return()) * 100.0
    except Exception:
        try:
            return float(pf.stats()["Total Return [%]"])
        except Exception:
            return np.nan


def _safe_sharpe(pf) -> float:
    for getter in (
        lambda: float(pf.sharpe_ratio()),
        lambda: float(pf.stats().get("Sharpe Ratio")),
        lambda: float(pf.stats().get("Sharpe Ratio ", np.nan)),
    ):
        try:
            val = getter()
            if np.isfinite(val):
                return val
        except Exception:
            pass

    # Fallback to trades Sharpe
    try:
        if hasattr(pf.trades, 'sharpe_ratio'):
            val = pf.trades.sharpe_ratio()
            if np.isfinite(val):
                return float(val)
    except Exception:
        pass

    # Custom fallback on trades PnL
    try:
        rec = pf.trades.records_readable
        if len(rec) > 0:
            pnl_col = "PnL" if "PnL" in rec.columns else "Pnl"
            if pnl_col in rec.columns:
                pnl = rec[pnl_col]
                if len(pnl) > 1 and pnl.std() > 0:
                    return sharpe(pnl / 100)  # assume daily returns from PnL %
    except Exception:
        pass

    # Fallback custom on portfolio returns
    try:
        returns = pf.returns.dropna()
        if len(returns) > 0 and returns.std() > 0:
            return sharpe(returns)
    except Exception:
        pass
    return np.nan


def _safe_max_dd(pf) -> float:
    try:
        return abs(float(pf.max_drawdown()) * 100.0)
    except Exception:
        try:
            return abs(float(pf.stats()["Max Drawdown [%]"]))
        except Exception:
            pass

    # Fallback to trades Max DD
    try:
        if hasattr(pf.trades, 'max_drawdown'):
            val = pf.trades.max_drawdown()
            if np.isfinite(val):
                return abs(float(val)) * 100.0
    except Exception:
        pass

    # Custom fallback on trades PnL
    try:
        rec = pf.trades.records_readable
        if len(rec) > 0:
            pnl_col = "PnL" if "PnL" in rec.columns else "Pnl"
            if pnl_col in rec.columns:
                pnl = rec[pnl_col]
                if len(pnl) > 0:
                    cum_pnl = pnl.cumsum()
                    drawdown = cum_pnl / cum_pnl.cummax() - 1
                    return abs(drawdown.min()) * 100.0
    except Exception:
        pass

    # Fallback custom on portfolio returns
    try:
        returns = pf.returns.dropna()
        if len(returns) > 0:
            cumrets = (1 + returns).cumprod()
            drawdown = cumrets / cumrets.cummax() - 1
            return abs(drawdown.min()) * 100.0
    except Exception:
        pass
    return np.nan


def _win_rate_and_trades(pf) -> Tuple[float, int]:
    trades_n = 0
    win_rate = np.nan
    try:
        rec = pf.trades.records_readable
        trades_n = len(rec)
        if trades_n > 0:
            pnl_col = "PnL" if "PnL" in rec.columns else ("Pnl" if "Pnl" in rec.columns else None)
            if pnl_col is not None:
                win_rate = (rec[pnl_col] > 0).mean() * 100.0
    except Exception:
        try:
            rec = pf.trades.records
            trades_n = len(rec)
            if trades_n > 0 and "pnl" in rec.dtype.names:
                win_rate = (rec["pnl"] > 0).mean() * 100.0
        except Exception:
            pass
    return float(win_rate), int(trades_n)


def run_backtest(
    close_wide: pd.DataFrame,
    signals: Dict[str, Dict[str, pd.Series]],
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    direction: str = "longonly",
    save_trades: bool = True,
    report_path: str | os.PathLike = "reports/summary_baseline.csv",
    # NOVO: tamanho de posição por barra (shares); DataFrame index=tempo, cols=tickers
    size_wide: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, "vbt.portfolio.base.Portfolio"]]:
    Path("reports").mkdir(parents=True, exist_ok=True)

    tickers = [t for t in close_wide.columns if t in signals]
    rows = []
    portfolios = {}

    for t in tickers:
        close = close_wide[t].dropna()
        if close.empty:
            continue

        e_raw = signals[t].get("entries", None)
        x_raw = signals[t].get("exits", None)

        e = _ensure_bool_series(e_raw, close.index).shift(1, fill_value=False)  # reforço do T+1
        x = _ensure_bool_series(x_raw, close.index).shift(1, fill_value=False)

        print(f"DEBUG {t}: entries={int(e.sum())} exits={int(x.sum())}")

        kwargs = dict(
            close=close,
            entries=e,
            exits=x,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            direction=direction,
            accumulate=True,
        )

        if size_wide is not None and t in size_wide.columns:
            # alinha e passa Series de shares por barra
            kwargs["size"] = size_wide[t].reindex(close.index).ffill().fillna(0.0)

        # from_signals com acumulação e (opcional) tamanho por barra
        pf = vbt.Portfolio.from_signals(
            close,
            entries=e,
            exits=x,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            freq="B",
            accumulate=True,
            size=(size_wide[t].reindex(close.index).ffill().fillna(0.0) if (size_wide is not None and t in size_wide.columns) else None),
        )

        portfolios[t] = pf

        total_ret = _safe_total_return(pf)
        sharpe = _safe_sharpe(pf)
        max_dd = _safe_max_dd(pf)
        win_rate, n_trades = _win_rate_and_trades(pf)

        rows.append(
            {
                "ticker": t,
                "Total Return [%]": total_ret,
                "Sharpe Ratio": sharpe,
                "Win Rate [%]": win_rate,
                "Max Drawdown [%]": max_dd,
                "Trades": n_trades,
            }
        )

        if save_trades:
            try:
                trades = pf.trades.records_readable
                trades.to_csv(Path("reports") / f"trades_{t}.csv", index=False)
            except Exception:
                pass

    summary_df = (
        pd.DataFrame.from_records(rows).set_index("ticker")
        if rows else
        pd.DataFrame(columns=["Total Return [%]", "Sharpe Ratio", "Win Rate [%]", "Max Drawdown [%]", "Trades"])
    )

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(report_path)

    return summary_df, portfolios
