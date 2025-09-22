# src/ingest.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

def get_prices(tickers, start="2015-01-01", end=None) -> pd.DataFrame:
    """
    Baixa OHLCV ajustado do yfinance.
    Retorna um DataFrame 'Close' com colunas = tickers.
    """
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by='ticker'
    )
    # yfinance pode retornar multiindex diferente quando 1 vs n tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs('Close', axis=1, level=1)
    else:
        close = data[['Close']].rename(columns={'Close': tickers if isinstance(tickers,str) else tickers[0]})
    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    return close.sort_index()
