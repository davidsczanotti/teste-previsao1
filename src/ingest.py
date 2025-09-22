# src/ingest.py
import pandas as pd
import time
from yahooquery import Ticker

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA"]


def get_prices(start="2015-01-01"):
    """
    Baixa os preços de fechamento para uma lista de tickers usando yahooquery,
    que é mais robusto contra falhas de API.
    """
    print("Baixando dados com yahooquery...")
    try:
        # yahooquery baixa todos os tickers de forma eficiente e robusta
        tickers = Ticker(TICKERS, asynchronous=True)
        df = tickers.history(start=start, adj_ohlc=True)

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Nenhum dado foi retornado pelo yahooquery.")

        # O resultado vem em um MultiIndex, vamos pivotar para o formato desejado
        close = df.reset_index().pivot(index="date", columns="symbol", values="close")
        print("Sucesso ao baixar os dados!")
        return close.dropna(how="all")
    except Exception as e:
        print(f"Ocorreu um erro com yahooquery: {e}")
        raise ConnectionError("Não foi possível baixar nenhum dado.")
