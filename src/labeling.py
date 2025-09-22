# src/labeling.py
import numpy as np
import pandas as pd
from vectorbt.signals.factory import SignalFactory


def triple_barrier_labels(close: pd.Series, pt_mult=2, sl_mult=2, t_verts_days=10):
    # Implementação simplificada do Triple Barrier usando VectorBT

    # Calcular a volatilidade diária (usando rolling std)
    # Acessa as funções do vectorbt através do acessor `.vbt`
    vol = close.vbt.rolling_std(window=20, min_periods=20)
    vol = vol.fillna(method="bfill")  # Preencher NaNs iniciais

    # Definir barreiras de take-profit e stop-loss baseadas na volatilidade
    # Retorno futuro para t_verts_days
    future_returns = close.pct_change(t_verts_days).shift(-t_verts_days)

    # Labels: 1 para alta (retorno > pt_mult * vol), -1 para baixa (retorno < -sl_mult * vol), 0 caso contrário
    labels = pd.DataFrame(index=close.index, columns=["bin"])
    labels["bin"] = 0
    labels.loc[future_returns > pt_mult * vol, "bin"] = 1
    labels.loc[future_returns < -sl_mult * vol, "bin"] = -1
    return labels.dropna()
