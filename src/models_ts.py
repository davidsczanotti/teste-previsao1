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
                        valid_tail: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Treina NHITS por ticker e prevê h passos à frente (horizonte multi-step).
    long_df: colunas ['unique_id','ds','y', ... (exógenas opcionais)]
    Retorna:
      - yhat_df: previsões (colunas: unique_id, ds, y_hat)
      - split_info: df com 'train_end' por ticker para separar o backtest
    """
    # Mantemos somente colunas mínimas para o baseline univariado
    df = long_df[['unique_id','ds','y']].dropna().copy()
    df = df.sort_values(['unique_id','ds'])

    # split simples: guarda o fim do treino por ticker
    split_info = (df.groupby('unique_id')['ds'].max()
                    .rename('train_end')
                    .reset_index())
    # Modelo
    model = NHITS(h=h, input_size=input_size, max_steps=max_steps, scaler_type='robust')
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=df)

    yhat_df = nf.predict()
    # yhat_df tem colunas: ['unique_id','ds','NHITS']
    yhat_df = yhat_df.rename(columns={'NHITS': 'y_hat'})
    return yhat_df, split_info
