# src/features.py
import pandas as pd
import vectorbt as vbt


def add_ta(df_close: pd.DataFrame) -> pd.DataFrame:
    # Exemplo: aplica RSI/MACD por ativo
    feats = {}
    for col in df_close.columns:
        # Usa vectorbt para calcular os indicadores
        rsi = vbt.RSI.run(df_close[col], window=14).rsi
        macd = vbt.MACD.run(df_close[col]).macd

        # Junta os resultados em um DataFrame
        f = pd.concat([df_close[col].rename("close"), rsi, macd], axis=1)
        feats[col] = f

    # concat por coluna com sufixo do ticker
    out = pd.concat(feats, axis=1)
    return out
