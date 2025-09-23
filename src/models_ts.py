# src/models_ts.py
from typing import Optional
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE


# Se você já tiver um helper _detect_pred_col definido, pode reaproveitar.
# Caso não tenha, este aqui resolve 99% dos casos.
def _detect_pred_col(columns, model_name_hint: str = "NHITS") -> str:
    cols = [str(c) for c in columns]
    # candidatos comuns
    for c in ("y_hat", "yhat", "y_pred", "prediction"):
        if c in cols:
            return c
    # às vezes o nome do modelo vira coluna
    if model_name_hint in cols:
        return model_name_hint
    # fallback: pega a única coluna "não meta"
    meta = {"unique_id", "id", "ds", "y", "cutoff", "index"}
    remainder = [c for c in cols if c not in meta]
    if len(remainder) == 1:
        return remainder[0]
    raise KeyError(f"Não encontrei coluna de previsão em: {columns}")


def train_predict_nhits(
    # compat de nome do argumento
    long_df: Optional[pd.DataFrame] = None,
    df_long: Optional[pd.DataFrame] = None,
    # Hiperparâmetros
    horizon: int = 5,
    input_size: int = 60,
    n_windows: int = 8,
    step_size: int = 1,
    max_steps: int = 300,
    seed: int = 1,
    start_padding_enabled: bool = True,
    # freq opcional
    freq: Optional[str] = None,
    # engole kwargs legados
    **kwargs,
) -> pd.DataFrame:
    """
    Treina NHITS em janelas (cross_validation) e retorna DataFrame com colunas:
    ['unique_id', 'ds', 'y_hat'].

    Aplica:
      - LEAD (ds > cutoff) para evitar look-ahead
      - dedup por (unique_id, ds) mantendo a previsão mais recente
    """
    # 0) entrada
    if long_df is None:
        long_df = df_long
    if long_df is None:
        raise ValueError("Passe o DataFrame via 'long_df' ou 'df_long' com colunas ['unique_id','ds','y'].")

    required_cols = {"unique_id", "ds", "y"}
    missing = required_cols - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df/df_long está sem as colunas obrigatórias: {missing}")

    df = long_df[["unique_id", "ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # 1) semente
    seed_everything(seed, workers=True)

    # 2) modelo
    nhits = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        start_padding_enabled=start_padding_enabled,
    )

    # 3) inferir freq se não vier
    if freq is None:
        try:
            # infere da série mais longa
            tmp = (
                df.sort_values("ds")
                .groupby("unique_id", group_keys=False)["ds"]
                .apply(lambda s: pd.infer_freq(s) or "")
            )
            freq = next((f for f in tmp.values if f), None) or "B"
        except Exception:
            freq = "B"

    nf = NeuralForecast(models=[nhits], freq=freq)

    # 4) cross-validation (walk-forward)
    fcst = nf.cross_validation(df=df, n_windows=n_windows, step_size=step_size)

    # 5) normalizar: garantir que unique_id / cutoff / ds sejam colunas
    index_names = list(getattr(fcst.index, "names", []))
    if any(name in index_names for name in ["unique_id", "cutoff", "ds"]):
        fcst = fcst.reset_index()

    # algumas versões usam 'id' no lugar de 'unique_id'
    if "unique_id" not in fcst.columns and "id" in fcst.columns:
        fcst = fcst.rename(columns={"id": "unique_id"})

    # 6) tipos de data
    if "ds" in fcst.columns:
        fcst["ds"] = pd.to_datetime(fcst["ds"])
    if "cutoff" in fcst.columns:
        fcst["cutoff"] = pd.to_datetime(fcst["cutoff"])

    # 7) detectar coluna de previsão
    pred_col = _detect_pred_col(fcst.columns, model_name_hint=nhits.__class__.__name__)

    # 8) aplicar LEAD: somente previsões estritamente após o cutoff
    if "cutoff" in fcst.columns:
        mask_lead = fcst["ds"] > fcst["cutoff"]
        # seleciona apenas colunas necessárias
        cols = ["unique_id", "ds", pred_col]  # cutoff não é necessário no output final
        fcst = fcst.loc[mask_lead, cols].copy()
    else:
        # fallback se não vier cutoff (raro em NF)
        fcst = fcst[["unique_id", "ds", pred_col]].copy()

    # 9) dedup por (unique_id, ds) mantendo a previsão mais recente
    before = len(fcst)
    fcst = fcst.sort_values(["unique_id", "ds"]).drop_duplicates(subset=["unique_id", "ds"], keep="last")
    removed = before - len(fcst)
    if removed > 0:
        print(f"[models_ts] Dedup com cutoff: removidas {removed} linhas duplicadas por (unique_id, ds).")

    # 10) saída padronizada
    out = fcst.rename(columns={pred_col: "y_hat"}).reset_index(drop=True)
    return out
