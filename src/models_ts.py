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
                        n_windows: int = 260,   # ~1 ano de janelas diárias
                        step_size: int = 1) -> pd.DataFrame:
    """
    Treina NHITS e gera previsões walk-forward (n_windows) com horizonte h.
    Retorna DataFrame com ['unique_id','ds','y_hat'] cobrindo várias janelas.
    """
    df = long_df[['unique_id','ds','y']].dropna().sort_values(['unique_id','ds']).copy()
    model = NHITS(h=h, input_size=input_size, max_steps=max_steps, scaler_type='robust')
    nf = NeuralForecast(models=[model], freq=freq)

    # Rolling forecasts (backtesting)
    try:
        fcst = nf.cross_validation(df=df, n_windows=n_windows, step_size=step_size, h=h)
    except Exception:
        # Fallback: última janela apenas (menos trades, mas não quebra)
        nf.fit(df=df)
        fcst = nf.predict()

    if 'unique_id' not in fcst.columns:
        fcst = fcst.reset_index()

    # renomeia coluna do modelo (ex.: 'NHITS') para 'y_hat'
    pred_cols = [c for c in fcst.columns if c not in ('unique_id','ds','y')]
    if len(pred_cols) >= 1:
        fcst = fcst.rename(columns={pred_cols[0]: 'y_hat'})
    else:
        # caso o pacote já inclua 'y_hat' direto
        if 'y_hat' not in fcst.columns:
            raise ValueError(f"Não achei coluna de previsão em {fcst.columns.tolist()}")

    return fcst[['unique_id','ds','y_hat']].sort_values(['unique_id','ds'])
