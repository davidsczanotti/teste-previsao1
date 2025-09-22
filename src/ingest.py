# src/ingest.py
import yfinance as yf
import pandas as pd
import time
import requests

TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA"]


def get_prices(start="2015-01-01"):
    """
    Baixa os preços de fechamento para uma lista de tickers, um de cada vez,
    com um longo intervalo para evitar bloqueios de API ("Too Many Requests").
    """
    # Configura uma sessão de requisições simples com um cabeçalho de navegador
    session = requests.Session()
    session.headers["User-agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )

    all_closes = []
    for ticker in TICKERS:
        try:
            print(f"Baixando dados para {ticker}...")
            data = yf.download(ticker, start=start, auto_adjust=True, progress=False, session=session)
            if data.empty:
                print(f"AVISO: Nenhum dado retornado para {ticker}.")
                continue

            all_closes.append(data["Close"].rename(ticker))
            print(f"Sucesso para {ticker}. Aguardando 30 segundos...")
            time.sleep(30)  # << A PAUSA ESTRATÉGICA
        except Exception as e:
            print(f"Falha ao baixar {ticker}: {e}")

    if not all_closes:
        raise ConnectionError("Não foi possível baixar nenhum dado.")

    return pd.concat(all_closes, axis=1).dropna(how="all")
