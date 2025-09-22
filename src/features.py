# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np
import ta  # lib "ta" (não 'pandas-ta')

def make_long_df(close: pd.DataFrame) -> pd.DataFrame:
    """
    Converte wide->long com colunas: ds (data), unique_id (ticker), y (preço).
    """
    df = close.copy()
    df = df.reset_index().melt(id_vars=df.index.name or 'Date', var_name='unique_id', value_name='y')
    df = df.rename(columns={df.columns[0]: 'ds'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df.dropna()

def add_ta_features(long_df: pd.DataFrame, window_rsi: int = 14) -> pd.DataFrame:
    """
    Calcula RSI e MACD por ticker e junta no long_df.
    """
    def _feat_grp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('ds').copy()
        # RSI
        g['rsi'] = ta.momentum.RSIIndicator(close=g['y'], window=window_rsi).rsi()
        # MACD
        macd = ta.trend.MACD(close=g['y'])
        g['macd'] = macd.macd()
        g['macd_signal'] = macd.macd_signal()
        g['macd_diff'] = macd.macd_diff()
        # retornos
        g['ret_1d'] = g['y'].pct_change()
        return g

    out = (long_df
           .groupby('unique_id', group_keys=False)
           .apply(_feat_grp))
    return out
