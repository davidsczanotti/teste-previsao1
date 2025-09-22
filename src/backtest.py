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
        "vectorbt não está instalado/compatível. " "Verifique sua versão (no seu projeto usa 0.28.1)."
    ) from e


# =========================
# Helpers de alinhamento
# =========================
def _ensure_bool_series(s: pd.Series | None, index: pd.Index) -> pd.Series:
    """Garante Series booleana, alinhada ao índice de preços (index)."""
    if s is None:
        return pd.Series(False, index=index)
    out = s.reindex(index, fill_value=False)
    if out.dtype != bool:
        out = out.astype(bool)
    return out


# =========================
# Helpers de métricas (sem warnings de freq)
# =========================
def _infer_periods_per_year(idx: pd.Index) -> int:
    """
    Tenta inferir períodos/ano a partir da frequência do índice.
    Padrão: 252 (dias úteis).
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return 252

    # tenta inferir freq
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None

    if freq is None:
        # fallback por densidade de datas: assume diário útil
        return 252

    freq = freq.upper()
    if freq.startswith("B") or freq.startswith("D"):
        return 252
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    if freq.startswith("A") or freq.startswith("Y"):
        return 1
    # fallback
    return 252


def _compute_total_return(returns: pd.Series) -> float:
    """Total return (%) a partir de retornos periódicos."""
    if returns.empty:
        return np.nan
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity.iloc[-1] - 1.0) * 100.0)


def _compute_sharpe(returns: pd.Series, periods_per_year: int, rf_per_period: float = 0.0) -> float:
    """
    Sharpe anualizado (excess return / std * sqrt(periods_per_year)).
    rf_per_period: taxa livre de risco por período (0 por simplicidade).
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    excess = r - rf_per_period
    mu = excess.mean()
    sigma = excess.std(ddof=0)
    if sigma == 0 or not np.isfinite(sigma):
        return np.nan
    return float(mu / sigma * np.sqrt(periods_per_year))


def _compute_max_drawdown(returns: pd.Series) -> float:
    """Max Drawdown (%) como valor positivo (magnitude)."""
    if returns.empty:
        return np.nan
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    dd = 1.0 - (curve / peak).clip(upper=1.0)
    mdd = dd.max()
    return float(mdd * 100.0)


def _win_rate_and_trades(pf) -> Tuple[float, int]:
    """
    Win Rate (%) e número de trades de forma robusta a diferenças de versão.
    """
    trades_n = 0
    win_rate = np.nan
    try:
        rec = pf.trades.records_readable  # DataFrame “legível”
        trades_n = len(rec)
        if trades_n > 0:
            pnl_col = "PnL" if "PnL" in rec.columns else ("Pnl" if "Pnl" in rec.columns else None)
            if pnl_col is not None:
                win_rate = (rec[pnl_col] > 0).mean() * 100.0
    except Exception:
        try:
            rec = pf.trades.records  # np.ndarray estruturado
            trades_n = len(rec)
            if trades_n > 0 and "pnl" in rec.dtype.names:
                win_rate = (rec["pnl"] > 0).mean() * 100.0
        except Exception:
            pass

    return float(win_rate), int(trades_n)


# =========================
# Função principal
# =========================
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
    Backtest por ticker aplicando EXECUÇÃO NO PRÓXIMO CANDLE (shift(1))
    nos sinais de entrada/saída, já alinhados ao índice de preços do ticker.

    Parameters
    ----------
    close_wide : DataFrame  (colunas = tickers, índice = datetime)
    signals    : { ticker: {"entries": Series[bool], "exits": Series[bool]} }
    init_cash, fees, slippage, direction : parâmetros do vectorbt
    save_trades : salva trades em reports/trades_{ticker}.csv
    report_path : caminho do CSV de resumo

    Returns
    -------
    summary_df : DataFrame métricas por ticker
    portfolios : {ticker: Portfolio}
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
        e = _ensure_bool_series(signals[t].get("entries"), close.index)
        x = _ensure_bool_series(signals[t].get("exits"), close.index)

        # 2) EXECUTAR NO PRÓXIMO CANDLE (evita look-ahead)
        e = e.shift(1, fill_value=False)
        x = x.shift(1, fill_value=False)

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
            accumulate=True,  # mantém posição até sinal de saída
        )
        portfolios[t] = pf

        # 4) métricas (manuais para evitar warnings de freq)
        periods_per_year = _infer_periods_per_year(close.index)
        returns = pf.returns()  # série de retornos por período

        total_ret = _compute_total_return(returns)
        sharpe = _compute_sharpe(returns, periods_per_year)
        max_dd = _compute_max_drawdown(returns)
        win_rate, n_trades = _win_rate_and_trades(pf)

        rows.append(
            {
                "ticker": t,
                "Total Return [%]": total_ret,
                "Sharpe Ratio": sharpe,
                "Win Rate [%]": win_rate,
                "Max Drawdown [%]": max_dd,  # positivo
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

    # resumo
    if rows:
        summary_df = pd.DataFrame.from_records(rows).set_index("ticker")
        summary_df = summary_df.sort_values("Total Return [%]", ascending=False)
    else:
        summary_df = pd.DataFrame(
            columns=["Total Return [%]", "Sharpe Ratio", "Win Rate [%]", "Max Drawdown [%]", "Trades"]
        )

    # salvar CSV
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(report_path)

    # print amigável
    pd.options.display.float_format = lambda x: f"{x:,.6f}"
    print("\n==== RESUMO ====\n")
    print(summary_df)
    print(f"\nRelatório salvo em: {report_path}")

    return summary_df, portfolios
