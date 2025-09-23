# src/signals.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _pivot_yhat_wide(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df is None or not isinstance(forecast_df, pd.DataFrame):
        raise ValueError("forecast_df precisa ser DataFrame com ['unique_id','ds',<yhat_col>].")

    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    if "unique_id" not in df.columns or "ds" not in df.columns:
        raise ValueError("forecast_df precisa conter 'unique_id' e 'ds'.")

    ycol = None
    for c in ("y_hat", "NHITS", "yhat", "forecast", "prediction", "y"):
        if c in df.columns:
            ycol = c
            break
    if ycol is None:
        raise ValueError(f"Coluna de previsão não encontrada. Colunas: {list(df.columns)}")

    wide = df.pivot(index="ds", columns="unique_id", values=ycol).sort_index()
    return wide


def build_signals_from_forecast(
    forecast_df: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    exp_thresh: float = 0.002,
    consec: int = 1,
    trend_sma: Optional[int] = None,
    only_non_overlapping: bool = True,
    # NOVO: limiar dinâmico
    dyn_thresh_k: Optional[float] = None,
    vol_window: int = 20,
    # logs
    debug: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, pd.Series]]:
    if close_wide is None or not isinstance(close_wide, pd.DataFrame):
        raise ValueError("'close_wide' precisa ser DataFrame (colunas=tickers, índice=datetime).")

    yhat_wide = _pivot_yhat_wide(forecast_df)
    yhat_wide = yhat_wide.reindex(close_wide.index)

    exp_ret = (yhat_wide - close_wide) / close_wide

    # --- limiar (fixo OU dinâmico) ---
    if dyn_thresh_k is not None:
        # vol diária realizada
        vol = close_wide.pct_change().rolling(window=vol_window, min_periods=vol_window).std()
        thr = (dyn_thresh_k * vol).clip(lower=1e-5)  # piso pequeno para não zerar
        entries = exp_ret > thr
        exits = exp_ret < -thr
    else:
        entries = exp_ret > exp_thresh
        exits = exp_ret < -exp_thresh

    if consec and consec > 1:
        entries = entries & (entries.rolling(window=consec, min_periods=consec).sum() >= consec)
        exits = exits & (exits.rolling(window=consec, min_periods=consec).sum() >= consec)

    if trend_sma and trend_sma > 0:
        sma = close_wide.rolling(window=int(trend_sma), min_periods=int(trend_sma)).mean()
        entries = entries & (close_wide > sma)
        # exits = exits | (close_wide < sma)  # opcional

    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    # Executar no próximo candle
    entries = entries.shift(1, fill_value=False)
    exits = exits.shift(1, fill_value=False)

    if only_non_overlapping:
        cleaned = {}
        for t in close_wide.columns:
            e = entries[t].to_numpy(copy=True)
            x = exits[t].to_numpy(copy=True)
            out = np.zeros_like(e, dtype=bool)
            in_pos = False
            for i in range(e.shape[0]):
                if not in_pos and e[i]:
                    out[i] = True
                    in_pos = True
                if in_pos and x[i]:
                    in_pos = False
            cleaned[t] = pd.Series(out, index=entries.index)
        entries = pd.DataFrame(cleaned)

    signals: Dict[str, Dict[str, pd.Series]] = {}
    for t in close_wide.columns:
        e = entries[t].astype(bool)
        x = exits[t].astype(bool)
        if debug:
            print(f"DEBUG {t}: entries={int(e.sum())} exits={int(x.sum())}")
        signals[t] = {"entries": e, "exits": x}

    return signals
