# src/models_ts.py
from __future__ import annotations

from typing import Optional
import pandas as pd
import pytorch_lightning as pl

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE


def _ensure_long_df(df_long: Optional[pd.DataFrame], long_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aceita df_long ou long_df para evitar 'quebra' de compatibilidade."""
    if df_long is None and long_df is not None:
        df_long = long_df
    if df_long is None:
        raise ValueError("É necessário fornecer df_long (ou long_df).")
    for col in ("unique_id", "ds", "y"):
        if col not in df_long.columns:
            raise ValueError(f"df_long precisa conter a coluna '{col}'. Colunas recebidas: {df_long.columns.tolist()}")
    out = df_long[["unique_id", "ds", "y"]].copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.dropna(subset=["y"])
    return out


def train_predict_nhits(
    df_long: Optional[pd.DataFrame] = None,
    *,
    # alias para compatibilidade antiga
    long_df: Optional[pd.DataFrame] = None,
    # hiperparâmetros principais
    horizon: int = 5,
    n_windows: int = 8,
    input_size: int = 60,
    max_steps: int = 300,
    random_seed: int = 1,
    # para séries curtas
    start_padding_enabled: bool = True,
    # tolera argumentos extras (p.ex. verbose, use_gpu, etc.)
    **_: object,
) -> pd.DataFrame:
    """
    Treina NHITS em validação-rolling (cross_validation) e retorna previsões no formato:
    [unique_id, ds, y_hat].

    Parâmetros mais usados:
      - horizon: passo de previsão (h)
      - n_windows: nº de janelas rolling
      - input_size: janela de entrada (lookback)
      - max_steps: passos de treino por janela
      - random_seed: seed global
      - start_padding_enabled: permite treinar com séries curtas (padding no início)
    """
    # 1) normalizar/validar df
    df = _ensure_long_df(df_long, long_df)

    # 2) seed para reprodutibilidade
    pl.seed_everything(random_seed, workers=True)

    # 3) instanciar modelo
    #    Observação: 'start_padding_enabled' é suportado no NHITS (na base _base_windows)
    nhits = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        random_seed=random_seed,
        start_padding_enabled=start_padding_enabled,
        # scaler_type padrão costuma ser "temporal" nas versões recentes
        scaler_type="temporal",
    )

    # freq='B' (dias úteis) funciona bem para cotações diárias
    nf = NeuralForecast(models=[nhits], freq="B")

    # 4) cross-validation (rolling). step_size=1 => re-treina/prevê diariamente
    try:
        fcst = nf.cross_validation(df=df, n_windows=n_windows, step_size=1)
    except Exception as e:
        # fallback amigável se a série estiver muito curta vs input_size
        msg = str(e)
        if "Time series is too short" in msg:
            # tenta com janela menor e padding habilitado
            nhits_small = NHITS(
                h=horizon,
                input_size=max(2, min(input_size, 10)),
                loss=MAE(),
                max_steps=max_steps,
                random_seed=random_seed,
                start_padding_enabled=True,
                scaler_type="temporal",
            )
            nf_small = NeuralForecast(models=[nhits_small], freq="B")
            fcst = nf_small.cross_validation(df=df, n_windows=max(1, n_windows), step_size=1)
        else:
            raise

    # 5) padronizar saída
    # cross_validation retorna colunas: ['unique_id','ds','y','cutoff','y_hat', ...]
    out = (
        fcst[["unique_id", "ds", "y_hat"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return out
