# src/models_ts.py
from __future__ import annotations
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

def train_predict_nhits(long_df: pd.DataFrame,
                        h: int = 5,
                        input_size: int = 60,
                        max_steps: int = 400,
                        freq: str = 'D') -> pd.DataFrame:
    """
    Treina NHITS e prevê h passos à frente.
    Retorna DataFrame com colunas: ['unique_id','ds','y_hat'].
    """
    df = long_df[['unique_id','ds','y']].dropna().sort_values(['unique_id','ds']).copy()

    model = NHITS(h=h, input_size=input_size, max_steps=max_steps, scaler_type='robust')
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=df)

    yhat_df = nf.predict()              # pode vir com unique_id no índice
    if 'unique_id' not in yhat_df.columns:
        yhat_df = yhat_df.reset_index() # garante a coluna

    # renomeia a coluna do modelo para 'y_hat'
    pred_cols = [c for c in yhat_df.columns if c not in ('unique_id','ds')]
    if len(pred_cols) != 1:
        raise ValueError(f"Esperava 1 coluna de previsão, recebi: {pred_cols}")
    yhat_df = yhat_df.rename(columns={pred_cols[0]: 'y_hat'})

    return yhat_df