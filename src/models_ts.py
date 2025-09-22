# src/models_ts.py
from __future__ import annotations
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

def train_predict_nhits(long_df: pd.DataFrame,
                        h: int = 5,
                        input_size: int = 60,
                        max_steps: int = 400,
                        freq: str = 'D',
                        n_windows: int = 260,
                        step_size: int = 1) -> pd.DataFrame:
    """
    Treina NHITS e gera previsões walk-forward (n_windows) com horizonte h.
    Retorna DataFrame com ['unique_id','ds','y_hat'].
    """
    df = long_df[['unique_id','ds','y']].dropna().sort_values(['unique_id','ds']).copy()

    model = NHITS(h=h, input_size=input_size, max_steps=max_steps, scaler_type='robust')
    nf = NeuralForecast(models=[model], freq=freq)

    # >>> TENTAR cross_validation e LOGAR qualquer falha
    try:
        fcst = nf.cross_validation(df=df, n_windows=n_windows, step_size=step_size)
    except Exception as e:
        print(f"[DEBUG] cross_validation falhou: {type(e).__name__}: {e}")
        raise  # reergue para vermos o stack completo

    # garantir unique_id como COLUNA
    if 'unique_id' not in fcst.columns:
        fcst = fcst.reset_index()

    # renomear a coluna do modelo para 'y_hat' e ignorar colunas auxiliares
    # cross_validation geralmente adiciona 'cutoff'
    cols_to_ignore = {'unique_id', 'ds', 'y', 'cutoff'}
    pred_cols = [c for c in fcst.columns if c not in cols_to_ignore]
    if len(pred_cols) != 1:
        raise ValueError(f"Esperava 1 coluna de previsão, recebi: {pred_cols}")
    fcst = fcst.rename(columns={pred_cols[0]: 'y_hat'})

    return fcst[['unique_id','ds','y_hat']].sort_values(['unique_id','ds'])
