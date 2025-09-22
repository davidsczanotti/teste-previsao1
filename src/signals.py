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
    Alinhamento robusto: usa merge_asof para mapear ds_pred -> ds_prev (pregão anterior).
    - exp_thresh: limiar fixo (ex.: 0.001 = 0,1%).
    - use_vol_threshold: se True, usa limiar dinâmico = vol_k * vol20.
    """
    # garante colunas
    if 'unique_id' not in yhat_df.columns:
        yhat_df = yhat_df.reset_index()
    pred_all = (yhat_df[['unique_id','ds','y_hat']]
                .rename(columns={'ds':'ds_pred'})
                .sort_values(['unique_id','ds_pred'])
                .copy())
    pred_all['ds_pred'] = pd.to_datetime(pred_all['ds_pred'])

    signals = {}

    for ticker in close_wide.columns:
        px = close_wide[ticker].dropna()
        df_px = pd.DataFrame({'ds_prev': px.index, 'close': px.values})
        df_px = df_px.sort_values('ds_prev')

        # previsões do ticker
        pred = pred_all[pred_all['unique_id'] == ticker][['ds_pred','y_hat']].copy()

        if pred.empty:
            # sem previsão => sem sinais
            signals[ticker] = {
                'entries': pd.Series(False, index=px.index),
                'exits':   pd.Series(False, index=px.index),
            }
            continue

        # mapear cada ds_pred ao pregão anterior (<= ds_pred)
        mapped = pd.merge_asof(
            pred.sort_values('ds_pred'),
            df_px[['ds_prev']].sort_values('ds_prev'),
            left_on='ds_pred', right_on='ds_prev',
            direction='backward',
            allow_exact_matches=False  # se ds_pred == ds_prev, exige anterior
        ).dropna(subset=['ds_prev'])

        # se várias previsões mapearem para o mesmo ds_prev (ex.: fim de semana),
        # agregue (pegar o maior y_hat costuma ser razoável para "exp_ret")
        mapped = (mapped.groupby('ds_prev', as_index=False)
                        .agg({'y_hat':'max'}))

        # série de expectativa no pregão anterior
        pred_on_prev = mapped.set_index('ds_prev')['y_hat'].reindex(px.index)

        # retorno esperado do D+1 vs preço atual em D
        exp_ret = pred_on_prev / px - 1.0

        if use_vol_threshold:
            vol20 = px.pct_change().rolling(vol_window).std()
            dyn_thresh = vol_k * vol20
            entries = (exp_ret > dyn_thresh).fillna(False)
        else:
            entries = (exp_ret > exp_thresh).fillna(False)

        exits = entries.shift(horizon).fillna(False)

        signals[ticker] = {
            'entries': entries.astype(bool),
            'exits':   exits.astype(bool),
        }

    return signals
