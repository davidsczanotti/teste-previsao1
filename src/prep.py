# src/prep.py
from __future__ import annotations

import pandas as pd


def prepare_long_and_features(close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o DataFrame wide de preços de fechamento (colunas = tickers, índice = datas)
    para o formato long esperado pelos modelos (unique_id, ds, y).

    Observações:
    - Mantém apenas a coluna alvo 'y' (preço). Se quiser, depois podemos
      anexar features artesanais aqui.
    - Remove NaNs em y.
    """
    if not isinstance(close_wide.index, pd.DatetimeIndex):
        close_wide = close_wide.copy()
        close_wide.index = pd.to_datetime(close_wide.index)

    # Garante índice ordenado
    close_wide = close_wide.sort_index()

    # Wide -> Long
    long_df = (
        close_wide
        .stack(dropna=False)            # empilha colunas em uma série multi-índice (data, ticker)
        .rename("y")
        .reset_index()                  # vira DataFrame
    )
    long_df.columns = ["ds", "unique_id", "y"]

    # Remove valores ausentes no alvo
    long_df = long_df.dropna(subset=["y"])

    # Ordena por ativo e data
    long_df = long_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return long_df
