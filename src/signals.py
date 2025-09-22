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
    min_hold: Union[int, Dict[str, int]] = 1,
    exit_symmetric: bool = False,
    # >>> NOVO: janela da SMA para filtro de tendência (int único ou dict por ticker)
    trend_sma: Union[int, Dict[str, int], None] = None,
) -> dict:
    """
    Sinais a partir da previsão do próximo pregão.
    - horizon e vol_k aceitam int/float único ou dict por ticker.
    - early_exit_on_flip: sai antes se a expectativa cair o bastante.
      (aqui usamos limiar simétrico: -vol_k * vol20 quando use_vol_threshold=True)
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

        mapped = pd.merge_asof(
            pred.sort_values("ds_pred"),
            df_px[["ds_prev"]].sort_values("ds_prev"),
            left_on="ds_pred",
            right_on="ds_prev",
            direction="backward",
            allow_exact_matches=False,
        ).dropna(subset=["ds_prev"])

        mapped = mapped.groupby("ds_prev", as_index=False).agg({"y_hat": "max"})
        pred_on_prev = mapped.set_index("ds_prev")["y_hat"].reindex(px.index)

        # expectativa de retorno p/ D+1
        exp_ret = pred_on_prev / px - 1.0

        # parâmetros por ticker
        hz = horizon[ticker] if isinstance(horizon, dict) else horizon
        vk = vol_k[ticker] if isinstance(vol_k, dict) else vol_k
        mh = min_hold[ticker] if isinstance(min_hold, dict) else min_hold
        tw = trend_sma.get(ticker) if isinstance(trend_sma, dict) else trend_sma

        # volatilidade 20d
        ret = px.pct_change()
        vol20 = ret.rolling(vol_window).std()

        # ENTRADA (mesmo que já estava)
        if use_vol_threshold:
            dyn_in = vk * vol20
            entries = ( (pred_on_prev / px - 1.0) > dyn_in ).fillna(False)
        else:
            entries = ( (pred_on_prev / px - 1.0) > exp_thresh ).fillna(False)

        # >>> NOVO: filtro de tendência (só entra se px > SMA(tw))
        if tw is not None and isinstance(tw, int) and tw > 1:
            sma = px.rolling(tw).mean()
            trend_ok = (px > sma)
            entries = (entries & trend_ok).fillna(False)

        # SAÍDA (igual ao que você já tem: horizonte OU flip < 0 depois de mh dias)
        exits_hz = entries.shift(hz).fillna(False)
        if early_exit_on_flip:
            if use_vol_threshold and exit_symmetric:
                dyn_out = -vk * vol20
                flip = ((pred_on_prev / px - 1.0) < dyn_out).shift(mh).fillna(False)
            else:
                flip = ((pred_on_prev / px - 1.0) < 0.0).shift(mh).fillna(False)
            exits = (exits_hz | flip)
        else:
            exits = exits_hz

        signals[ticker] = {'entries': entries.astype(bool), 'exits': exits.astype(bool)}

    return signals
