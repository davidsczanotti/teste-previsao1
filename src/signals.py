# src/signals.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union, Dict


def build_signals_from_forecast(
    close_wide: pd.DataFrame,
    yhat_df: pd.DataFrame,
    horizon: Union[int, Dict[str, int]] = 5,
    exp_thresh: float = 0.001,
    use_vol_threshold: bool = False,
    vol_window: int = 20,
    vol_k: Union[float, Dict[str, float]] = 0.25,
    early_exit_on_flip: bool = True,
    min_hold: int = 1,
) -> dict:
    """
    Constrói sinais a partir das previsões do próximo pregão.
    - horizon: int único ou dict por ticker (ex.: {"VALE3.SA":5, "PETR4.SA":3})
    - vol_k:   float único ou dict por ticker
    - early_exit_on_flip: se True, sai antes do horizon quando exp_ret < 0
    - min_hold: mínimo de pregões a segurar antes de permitir saída antecipada
    """
    if "unique_id" not in yhat_df.columns:
        yhat_df = yhat_df.reset_index()

    pred_all = (
        yhat_df[["unique_id", "ds", "y_hat"]]
        .rename(columns={"ds": "ds_pred"})
        .sort_values(["unique_id", "ds_pred"])
        .copy()
    )
    pred_all["ds_pred"] = pd.to_datetime(pred_all["ds_pred"])

    signals = {}

    for ticker in close_wide.columns:
        px = close_wide[ticker].dropna()
        df_px = pd.DataFrame({"ds_prev": px.index, "close": px.values}).sort_values("ds_prev")

        pred = pred_all[pred_all["unique_id"] == ticker][["ds_pred", "y_hat"]].copy()
        if pred.empty:
            signals[ticker] = {
                "entries": pd.Series(False, index=px.index),
                "exits": pd.Series(False, index=px.index),
            }
            continue

        # mapear previsão futura para o pregão anterior existente
        mapped = pd.merge_asof(
            pred.sort_values("ds_pred"),
            df_px[["ds_prev"]].sort_values("ds_prev"),
            left_on="ds_pred",
            right_on="ds_prev",
            direction="backward",
            allow_exact_matches=False,
        ).dropna(subset=["ds_prev"])

        # agrega previsões que caem no mesmo pregão anterior (fim de semana, etc.)
        mapped = mapped.groupby("ds_prev", as_index=False).agg({"y_hat": "max"})

        pred_on_prev = mapped.set_index("ds_prev")["y_hat"].reindex(px.index)
        exp_ret = pred_on_prev / px - 1.0  # expectativa p/ D+1 vs preço em D

        # parâmetros por ticker
        hz = horizon[ticker] if isinstance(horizon, dict) else horizon
        vk = vol_k[ticker] if isinstance(vol_k, dict) else vol_k

        # regra de entrada
        if use_vol_threshold:
            vol20 = px.pct_change().rolling(vol_window).std()
            dyn_thresh = vk * vol20
            entries = (exp_ret > dyn_thresh).fillna(False)
        else:
            entries = (exp_ret > exp_thresh).fillna(False)

        # regra de saída: horizonte OU virada negativa da expectativa
        exits_hz = entries.shift(hz).fillna(False)

        if early_exit_on_flip:
            # flip: expectativa negativa (abaixo de 0) – após min_hold
            flip = (exp_ret < 0.0).shift(min_hold).fillna(False)
            exits = exits_hz | flip
        else:
            exits = exits_hz

        signals[ticker] = {
            "entries": entries.astype(bool),
            "exits": exits.astype(bool),
        }

    return signals
