# src/signals.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


__all__ = ["build_signals_from_forecast"]


def _pivot_forecast_to_wide(forecast_df: pd.DataFrame, close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Espera forecast_df com colunas: ['unique_id', 'ds', 'y_hat'] (long).
    Retorna DataFrame wide indexado por ds e colunas = tickers, reindexado
    para o mesmo índice do close_wide (pode introduzir NaN onde não há previsão).
    """
    if not {"unique_id", "ds", "y_hat"} <= set(forecast_df.columns):
        raise ValueError("forecast_df deve conter colunas ['unique_id','ds','y_hat'].")

    wide = forecast_df.pivot(index="ds", columns="unique_id", values="y_hat")
    # manter apenas tickers presentes em close_wide e na mesma ordem
    cols = [c for c in close_wide.columns if c in wide.columns]
    wide = wide.reindex(columns=cols)
    # alinhar datas ao índice de preços
    wide = wide.reindex(close_wide.index)
    return wide


def _expected_return(yhat_wide: pd.DataFrame, close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Retorno esperado simples: (preço_previsto / preço_atual) - 1
    Mesma forma/índice/colunas entre yhat_wide e close_wide.
    """
    return (yhat_wide / close_wide) - 1.0


def _dynamic_threshold(close_wide: pd.DataFrame, k: float, vol_window: int) -> pd.DataFrame:
    """
    Limiar dinâmico por volatilidade: k * std(returns rolling).
    Sem look-ahead (usa apenas dados até t).
    """
    rets = close_wide.pct_change()
    vol = rets.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()
    return k * vol


def _require_consecutive(flags: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Exige 'n' sinais consecutivos True (por coluna). Mantém alinhamento.
    """
    if n <= 1:
        return flags
    as_int = flags.astype(int)
    consec_ok = as_int.rolling(n, min_periods=n).sum().eq(n)
    return consec_ok.reindex_like(flags).fillna(False)


def build_signals_from_forecast(
    forecast_df: pd.DataFrame,
    close_wide: pd.DataFrame,
    *,
    horizon: int = 1,  # mantido para compat; não usamos diretamente aqui
    exp_thresh: Optional[float] = None,
    dyn_thresh_k: Optional[float] = None,
    vol_window: int = 20,
    consec: int = 1,
    trend_sma: Optional[int] = None,
    # Filtros RSI opcionais
    rsi_window: Optional[int] = None,
    rsi_min: Optional[float] = None,
    # Evitar entradas sobrepostas enquanto posição está aberta
    only_non_overlapping: bool = False,
    debug: bool = False,
    **kwargs,  # engole kwargs extras vindos da config/CLI
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Constrói sinais a partir das previsões (long DataFrame -> wide),
    calcula retorno esperado por dia/ativo e aplica regras de entrada/saída.

    Retorna:
      { ticker: {"entries": Series[bool], "exits": Series[bool]} }
    """

    # 1) Previsão em wide e alinhada com o índice de preços
    yhat_wide = _pivot_forecast_to_wide(forecast_df, close_wide)

    # 2) Retorno esperado (mesmo índice/colunas que close_wide)
    exp_ret = _expected_return(yhat_wide, close_wide)

    # 3) Threshold: fixo ou dinâmico
    if dyn_thresh_k is not None:
        thr = _dynamic_threshold(close_wide, dyn_thresh_k, vol_window)
        # Para dias iniciais sem vol, evita operar
        thr = thr.fillna(np.inf)
    else:
        if exp_thresh is None:
            exp_thresh = 0.0
        thr = pd.DataFrame(exp_thresh, index=close_wide.index, columns=close_wide.columns)

    # 4) *** ALINHAR *** threshold ao exp_ret (corrige o erro de labels diferentes)
    thr = thr.reindex(index=exp_ret.index, columns=exp_ret.columns)

    # 5) Regras básicas (long-only): entra se exp_ret > +thr, sai se exp_ret < -thr
    entries = exp_ret > thr
    exits = exp_ret < -thr

    # 6) Filtro de tendência (SMA): só entra quando preço acima da SMA
    if trend_sma is not None and trend_sma > 1:
        sma = close_wide.rolling(trend_sma, min_periods=trend_sma).mean()
        sma = sma.reindex_like(exp_ret)
        trend_ok = (close_wide.reindex_like(exp_ret) > sma).fillna(False)
        entries = entries & trend_ok

    # 6.1) Filtro RSI (se habilitado): exige RSI >= rsi_min
    if rsi_window is not None and rsi_window > 1 and rsi_min is not None:
        try:
            import ta
        except Exception:
            raise RuntimeError("Pacote 'ta' é necessário para usar o filtro RSI.")

        rsi_cols = {}
        for t in close_wide.columns:
            s = close_wide[t].astype(float)
            r = ta.momentum.RSIIndicator(close=s, window=int(rsi_window)).rsi()
            rsi_cols[t] = r
        rsi_wide = pd.DataFrame(rsi_cols).reindex_like(exp_ret)
        rsi_ok = (rsi_wide >= float(rsi_min)).fillna(False)
        entries = entries & rsi_ok

    # 7) Exigir 'consec' dias consecutivos de sinal de entrada
    if consec and consec > 1:
        entries = _require_consecutive(entries.fillna(False), consec)

    # 8) Limpeza fina
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # 8.1) Suprimir entradas sobrepostas (uma posição por vez)
    if only_non_overlapping:
        pruned_entries = {}
        for t in close_wide.columns:
            e = entries.get(t, pd.Series(False, index=exp_ret.index)).copy()
            x = exits.get(t, pd.Series(False, index=exp_ret.index)).copy()
            # varredura temporal para impedir múltiplas entradas antes de uma saída
            open_pos = False
            e_out = []
            for ts in e.index:
                if x.loc[ts] and open_pos:
                    open_pos = False
                if e.loc[ts] and not open_pos:
                    e_out.append(True)
                    open_pos = True
                else:
                    e_out.append(False)
            pruned_entries[t] = pd.Series(e_out, index=e.index)
        entries = pd.DataFrame(pruned_entries).reindex_like(entries).fillna(False)

    # 9) Montar dict final, **alinhado ao índice de cada ticker** em close_wide
    signals: Dict[str, Dict[str, pd.Series]] = {}
    for t in close_wide.columns:
        e = entries.get(t, pd.Series(False, index=exp_ret.index)).reindex(close_wide.index, fill_value=False)
        x = exits.get(t, pd.Series(False, index=exp_ret.index)).reindex(close_wide.index, fill_value=False)
        signals[t] = {"entries": e.astype(bool), "exits": x.astype(bool)}

        if debug:
            print(f"[signals] {t}: entries={int(e.sum())} exits={int(x.sum())}")

    return signals
