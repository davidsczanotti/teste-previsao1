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
        "Verifique sua versão (no projeto usamos 0.28.x)."
    ) from e


# ==========================
# Utilidades internas
# ==========================

def _ensure_bool_series(s: pd.Series | None, index: pd.Index) -> pd.Series:
    """Garante Series booleana alinhada ao índice fornecido."""
    if s is None:
        return pd.Series(False, index=index)
    out = s.reindex(index, fill_value=False)
    if out.dtype != bool:
        out = out.astype(bool)
    return out


def _safe_total_return(pf) -> float:
    """Total Return em % (robusto entre versões)."""
    # 1) API moderna
    try:
        val = pf.total_return()
        val = float(val)  # fração (ex.: 0.34)
        return val * 100.0
    except Exception:
        pass
    # 2) via stats
    try:
        stats = pf.stats(freq="D")
        # pode vir como chave numérica ou string
        for k in ("Total Return [%]", "Total Return"):
            if k in stats:
                v = float(stats[k])
                # Se for fração (0.34), transforma em %
                return v * 100.0 if abs(v) <= 1.0 else v
    except Exception:
        pass
    return np.nan


def _safe_sharpe(pf) -> float:
    """Sharpe Ratio (robusto, com freq diária para evitar warnings)."""
    # 1) API direta
    for getter in (
        lambda: float(pf.sharpe_ratio(freq="D")),
        lambda: float(pf.sharpe_ratio()),  # fallback
    ):
        try:
            v = getter()
            if np.isfinite(v):
                return v
        except Exception:
            pass
    # 2) via stats
    try:
        stats = pf.stats(freq="D")
        for k in ("Sharpe Ratio", "Sharpe Ratio "):
            if k in stats and np.isfinite(stats[k]):
                return float(stats[k])
    except Exception:
        pass
    return np.nan


def _safe_sortino(pf) -> float:
    """Sortino Ratio (opcional, usado se desejar estender o relatório)."""
    try:
        return float(pf.sortino_ratio(freq="D"))
    except Exception:
        try:
            stats = pf.stats(freq="D")
            for k in ("Sortino Ratio", "Sortino Ratio "):
                if k in stats and np.isfinite(stats[k]):
                    return float(stats[k])
        except Exception:
            pass
    return np.nan


def _safe_max_dd(pf) -> float:
    """Max Drawdown como percentual **positivo** (robusto entre versões)."""
    # 1) API direta
    try:
        val = pf.max_drawdown()  # pode vir como fração positiva (0.2) ou negativa
        val = float(val)
        # normaliza para percentual positivo
        if abs(val) <= 1.0:
            return abs(val) * 100.0
        return abs(val)
    except Exception:
        pass
    # 2) via stats
    try:
        stats = pf.stats(freq="D")
        for k in ("Max Drawdown [%]", "Max Drawdown"):
            if k in stats:
                v = float(stats[k])
                # normaliza para percentual positivo
                if abs(v) <= 1.0:
                    return abs(v) * 100.0
                return abs(v)
    except Exception:
        pass
    return np.nan


def _win_rate_and_trades(pf) -> Tuple[float, int]:
    """Win rate (%) e nº de trades, lidando com diferenças de schema."""
    trades_n = 0
    win_rate = np.nan

    # 1) Tabela legível
    try:
        rec = pf.trades.records_readable  # DataFrame
        trades_n = int(len(rec))
        if trades_n > 0:
            pnl_col = None
            if "PnL" in rec.columns:
                pnl_col = "PnL"
            elif "Pnl" in rec.columns:
                pnl_col = "Pnl"
            if pnl_col:
                win_rate = float((rec[pnl_col] > 0).mean() * 100.0)
    except Exception:
        pass

    # 2) Fallback: ndarray estruturado
    if not np.isfinite(win_rate):
        try:
            rec = pf.trades.records  # np structured array
            trades_n = int(len(rec))
            if trades_n > 0 and "pnl" in rec.dtype.names:
                win_rate = float((rec["pnl"] > 0).mean() * 100.0)
        except Exception:
            pass

    return (win_rate if np.isfinite(win_rate) else np.nan), trades_n


# ==========================
# Função principal
# ==========================

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
    portfolios: Dict[str, vbt.portfolio.base.Portfolio] = {}

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

        # (debug leve, uma única linha por ticker)
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
                "Max Drawdown [%]": max_dd,  # sempre positivo
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
