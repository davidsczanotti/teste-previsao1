# src/signals.py
from __future__ import annotations
import pandas as pd
import numpy as np

def build_signals_from_forecast(close_wide: pd.DataFrame,
                                yhat_df: pd.DataFrame,
                                horizon: int = 5,
                                exp_thresh: float = 0.001,
                                use_vol_threshold: bool = False,
                                vol_window: int = 20,
                                vol_k: float = 0.25) -> dict:
    """
    Constrói sinais a partir das previsões de preço do próximo pregão.
    - exp_thresh: limiar fixo (ex.: 0.001 = 0,1%).
    - use_vol_threshold: se True, usa limiar dinâmico = vol_k * vol20.
    """
    if 'unique_id' not in yhat_df.columns:
        yhat_df = yhat_df.reset_index()
    yhat_df = yhat_df[['unique_id','ds','y_hat']].copy()
    yhat_df['ds'] = pd.to_datetime(yhat_df['ds'])

    signals = {}

    for ticker in close_wide.columns:
        px = close_wide[ticker].dropna()
        ret = px.pct_change()
        vol20 = ret.rolling(vol_window).std()

        pred = (yhat_df[yhat_df['unique_id'] == ticker]
                .set_index('ds')
                .sort_index())['y_hat']

        # Mapear cada previsão (para ds futuro) ao pregão anterior existente
        pred_on_prev = pd.Series(index=px.index, dtype='float64')
        # para evitar sobrescrever em sextas com previsões de sábado/domingo,
        # guardamos a MAIOR expectativa do par (Friday <- Sat/Sun)
        stash = {}

        for ds_pred, val in pred.items():
            loc = px.index.searchsorted(ds_pred) - 1
            if loc >= 0:
                ds_prev = px.index[loc]
                # manter o maior y_hat caso haja múltiplos mapeando ao mesmo ds_prev
                stash[ds_prev] = max(val, stash.get(ds_prev, -np.inf))

        if stash:
            pred_on_prev.loc[list(stash.keys())] = list(stash.values())

        exp_ret = pred_on_prev / px - 1.0

        if use_vol_threshold:
            dyn_thresh = vol_k * vol20
            entries = (exp_ret > dyn_thresh).fillna(False)
        else:
            entries = (exp_ret > exp_thresh).fillna(False)

        exits = entries.shift(horizon).fillna(False)
        signals[ticker] = {'entries': entries.astype(bool), 'exits': exits.astype(bool)}

    return signals
