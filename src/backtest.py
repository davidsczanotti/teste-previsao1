# src/backtest.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except Exception as e:
    raise RuntimeError(
        "vectorbt não está instalado/compatível. "
        "Verifique sua versão (no seu projeto usa 0.28.1)."
    ) from e


def _ensure_bool_series(s: pd.Series, index: pd.Index) -> pd.Series:
    """
    Garante Series booleana, alinhada ao índice de preços (index),
    preenchendo ausências com False.
    """
    if s is None:
        return pd.Series(False, index=index)
    out = s.reindex(index, fill_value=False)
    # alguns pipelines podem produzir dtype object/int; forçamos bool clean
    if out.dtype != bool:
        out = out.astype(bool)
    return out


def _safe_total_return(pf) -> float:
    """Retorna Total Return em % de forma robusta a diferenças de versão."""
    try:
        return float(pf.total_return()) * 100.0
    except Exception:
        try:
            return float(pf.stats()["Total Return [%]"])
        except Exception:
            return np.nan


def _safe_sharpe(pf) -> float:
    """Retorna Sharpe Ratio de forma robusta a diferenças de versão."""
    for getter in (
        lambda: float(pf.sharpe_ratio()),
        lambda: float(pf.stats().get("Sharpe Ratio")),
        lambda: float(pf.stats().get("Sharpe Ratio ", np.nan)),
    ):
        try:
            val = getter()
            if np.isfinite(val):
                return val
        except Exception:
            pass
    return np.nan


def _safe_max_dd(pf) -> float:
    """Retorna Max Drawdown em % de forma robusta."""
    try:
        return float(pf.max_drawdown()) * 100.0
    except Exception:
        try:
            return float(pf.stats()["Max Drawdown [%]"])
        except Exception:
            return np.nan


def _win_rate_and_trades(pf) -> Tuple[float, int]:
    """
    Calcula Win Rate (%) e número de trades a partir dos registros do vectorbt,
    lidando com diferenças de nomes de colunas (PnL / Pnl).
    """
    trades_n = 0
    win_rate = np.nan

    try:
        rec = pf.trades.records_readable  # DataFrame "bonitinho"
        trades_n = len(rec)
        if trades_n > 0:
            pnl_col = "PnL" if "PnL" in rec.columns else ("Pnl" if "Pnl" in rec.columns else None)
            if pnl_col is not None:
                win_rate = (rec[pnl_col] > 0).mean() * 100.0
    except Exception:
        # fallback: usa .records cru
        try:
            rec = pf.trades.records  # np.ndarray estruturado
            trades_n = len(rec)
            # algumas versões têm campo 'pnl'
            if trades_n > 0 and "pnl" in rec.dtype.names:
                win_rate = (rec["pnl"] > 0).mean() * 100.0
        except Exception:
            pass

    return float(win_rate), int(trades_n)


def run_backtest(
    close_wide: pd.DataFrame,
    signals: Dict[str, Dict[str, pd.Series]],
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    direction: str = "longonly",
    save_trades: bool = True,
    report_path: str | os.PathLike = "reports/summary_baseline.csv",
) -> Tuple[pd.DataFrame, Dict[str, "vbt.portfolio.base.Portfolio"]]:
    """
    Roda o backtest por ticker aplicando o *EXECUTE-NO-PRÓXIMO-CANDLE* (shift(1))
    nos sinais de entrada/saída, já alinhados ao índice de preços do ticker.

    Parameters
    ----------
    close_wide : DataFrame
        Preços de fechamento em colunas (tickers) e índice temporal.
    signals : dict
        { ticker: {"entries": Series[bool], "exits": Series[bool]} }
        As séries podem ter índice diferente do de preços; aqui alinhamos.
    init_cash, fees, slippage, direction : parâmetros do vectorbt
    save_trades : salva trades individuais em reports/trades_{ticker}.csv
    report_path : caminho do CSV de resumo

    Returns
    -------
    summary_df : DataFrame com métricas agregadas por ticker
    portfolios : dict {ticker: Portfolio}
    """
    Path("reports").mkdir(parents=True, exist_ok=True)

    tickers = [t for t in close_wide.columns if t in signals]
    rows = []
    portfolios = {}

    for t in tickers:
        close = close_wide[t].dropna()
        if close.empty:
            continue

        # 1) alinhar sinais ao índice de preços
        e_raw = signals[t].get("entries", None)
        x_raw = signals[t].get("exits", None)

        e = _ensure_bool_series(e_raw, close.index)
        x = _ensure_bool_series(x_raw, close.index)

        # 2) *** EXECUTAR NO PRÓXIMO CANDLE ***
        # (evita look-ahead: o sinal gerado em t só é operado em t+1)
        e = e.shift(1, fill_value=False)
        x = x.shift(1, fill_value=False)

        # (debug leve similar aos logs que você já imprime)
        print(f"DEBUG {t}: entries={int(e.sum())} exits={int(x.sum())}")

        # 3) construir Portfólio
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=e,
            exits=x,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            direction=direction,
            accumulate=True,   # mantém posição enquanto não sair
        )
        portfolios[t] = pf

        # 4) métricas
        total_ret = _safe_total_return(pf)
        sharpe = _safe_sharpe(pf)
        max_dd = _safe_max_dd(pf)
        win_rate, n_trades = _win_rate_and_trades(pf)

        rows.append(
            {
                "ticker": t,
                "Total Return [%]": total_ret,
                "Sharpe Ratio": sharpe,
                "Win Rate [%]": win_rate,
                "Max Drawdown [%]": max_dd,
                "Trades": n_trades,
            }
        )

        # 5) salvar trades por ticker (opcional)
        if save_trades:
            try:
                trades = pf.trades.records_readable
                trades.to_csv(Path("reports") / f"trades_{t}.csv", index=False)
            except Exception:
                pass

    # montar resumo
    if rows:
        summary_df = pd.DataFrame.from_records(rows).set_index("ticker")
    else:
        summary_df = pd.DataFrame(
            columns=["Total Return [%]", "Sharpe Ratio", "Win Rate [%]", "Max Drawdown [%]", "Trades"]
        )

    # salvar
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(report_path)

    return summary_df, portfolios
