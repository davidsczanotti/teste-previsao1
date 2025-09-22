# src/ingest.py
import yfinance as yf
import pandas as pd
import time
import requests_cache

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA"]


def get_prices(start="2015-01-01"):
    for i in range(3):  # Tenta baixar 3 vezes
        try:
            # Simula um navegador para evitar bloqueios e usa cache para acelerar downloads repetidos
            session = requests_cache.CachedSession("yfinance.cache")
            session.headers["User-agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            )

            data = yf.download(TICKERS, start=start, auto_adjust=True, progress=False, session=session)

            if data.empty:
                raise ValueError("Nenhum dado foi baixado.")

            # MultiIndex (coluna nivel 0 = campo OHLCV, nivel 1 = ticker)
            close = data["Close"].dropna(how="all")
            if not close.empty:
                return close  # DataFrame: dates x tickers
        except Exception as e:
            print(f"Tentativa {i+1} falhou: {e}")
            time.sleep(2)  # Espera 2 segundos antes de tentar de novo

    raise ConnectionError("Não foi possível baixar os dados após várias tentativas.")
