# src/models_ts.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS


def _infer_min_len_per_id(df: pd.DataFrame) -> int:
    """Menor quantidade de pontos entre os unique_id da base."""
    return int(df.groupby("unique_id")["ds"].nunique().min())


def _pick_input_size(min_len: int, h: int, default_cap: int = 64) -> int:
    """
    Define um input_size seguro:
    - pelo menos 2*h (regra prática para janelas)
    - no máximo 'default_cap'
    - não maior do que min_len - h (quando possível)
    """
    base = max(2 * h, 2)  # nunca < 2
    if min_len <= base + h:
        # séries muito curtas: deixa o mínimo e confia no start_padding_enabled
        return base
    return int(min(default_cap, max(base, min_len - h)))


def _coerce_forecast_df(df_fcst: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o DataFrame de saída do NeuralForecast para colunas:
    ['unique_id', 'ds', 'y_hat'].
    """
    cols = df_fcst.columns.str.lower().tolist()
    rename_map = {}
    for a, b in [
        ("unique_id", "unique_id"),
        ("ds", "ds"),
        ("y_hat", "y_hat"),
    ]:
        if a not in cols:
            # tenta achar variações
            for cand in df_fcst.columns:
                if cand.lower() == a:
                    rename_map[cand] = b
        else:
            # já existe com esse nome (ou variação), não faz nada
            pass
    if rename_map:
        df_fcst = df_fcst.rename(columns=rename_map)

    # às vezes vem com coluna do nome do modelo; se existir, descarta
    for extra in ["model", "Model"]:
        if extra in df_fcst.columns:
            df_fcst = df_fcst.drop(columns=[extra])

    # garante apenas as 3 colunas
    return df_fcst[["unique_id", "ds", "y_hat"]].copy()


def train_predict_nhits(
    df_long: pd.DataFrame,
    horizon: int,
    max_steps: int = 300,
    n_windows: int = 3,
    step_size: Optional[int] = None,
    learning_rate: float = 1e-3,
    input_size: Optional[int] = None,
    random_seed: int = 1,
    freq: str = "B",
) -> pd.DataFrame:
    """
    Treina NHITS em rolling (cross_validation) e retorna previsões em formato
    ['unique_id','ds','y_hat'].

    - Usa start_padding_enabled=True para lidar com séries curtas.
    - Ajusta input_size de forma automática se não for informado.
    - Tem um fallback: se o cross_validation falhar, faz fit+predict.
    """
    if step_size is None:
        step_size = horizon

    # Normalizações básicas de tipos/ordem
    df = df_long.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    min_len = _infer_min_len_per_id(df)
    if input_size is None:
        input_size = _pick_input_size(min_len=min_len, h=horizon)

    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        learning_rate=learning_rate,
        random_seed=random_seed,
        # **CHAVE** para séries curtas:
        start_padding_enabled=True,
    )

    nf = NeuralForecast(models=[model], freq=freq)

    # Tenta rolling CV primeiro (in-sample rolling); se falhar, faz fit+predict
    try:
        fcst = nf.cross_validation(df=df, n_windows=n_windows, step_size=step_size)
        # Em algumas versões vem com a coluna do nome do modelo. Normaliza:
        fcst = _coerce_forecast_df(fcst)
        return fcst
    except Exception as e:
        print(f"[DEBUG] cross_validation falhou: {type(e).__name__}: {e}")
        # fallback: treina uma vez e prevê (next-h passos à frente)
        nf.fit(df=df)
        fcst = nf.predict()  # horizonte fora da amostra
        fcst = _coerce_forecast_df(fcst)
        return fcst
