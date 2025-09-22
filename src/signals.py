# src/signals.py (substitua a função por esta)

from __future__ import annotations
import pandas as pd
import numpy as np

def build_signals_from_forecast(close_wide: pd.DataFrame,
                                yhat_df: pd.DataFrame,
                                horizon: int = 5,
                                exp_thresh: float = 0.003) -> dict:
    """
    Constrói sinais (entries/exits) por ticker a partir da previsão de preços.
    Lógica: usa a previsão de preço do próximo pregão (ds_pred) e a atribui
    ao pregão anterior (ds_prev) para decidir a entrada.
    """
    # garante colunas
    if 'unique_id' not in yhat_df.columns:
        yhat_df = yhat_df.reset_index()
    yhat_df = yhat_df[['unique_id','ds','y_hat']].copy()
    yhat_df['ds'] = pd.to_datetime(yhat_df['ds'])

    signals = {}

    for ticker in close_wide.columns:
        px = close_wide[ticker].dropna()
        pred = (yhat_df[yhat_df['unique_id'] == ticker]
                .set_index('ds')
                .sort_index())['y_hat']

        # mapeia cada ds_pred -> ds_prev (pregão anterior existente em px.index)
        pred_on_prev = pd.Series(index=px.index, dtype='float64')
        for ds_pred, val in pred.items():
            # posição onde ds_pred seria inserido em px.index
            loc = px.index.searchsorted(ds_pred) - 1
            if loc >= 0:
                ds_prev = px.index[loc]
                pred_on_prev.loc[ds_prev] = val

        exp_ret = pred_on_prev / px - 1.0
        entries = (exp_ret > exp_thresh).fillna(False)
        exits = entries.shift(horizon).fillna(False)

        signals[ticker] = {
            'entries': entries.astype(bool),
            'exits'  : exits.astype(bool)
        }

    return signals
