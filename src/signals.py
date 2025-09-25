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
    bb_window: Optional[int] = None,
    bb_k: float = 2.0,
    atr_window: Optional[int] = None,
    atr_stop_k: Optional[float] = None,
    cooldown_bars: int = 0,
    max_hold_bars: Optional[int] = None,
    # Evitar entradas sobrepostas enquanto posição está aberta
    only_non_overlapping: bool = False,
    debug: bool = False,
    **kwargs,  # engole kwargs extras vindos da config/CLI
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Constrói sinais a partir das previsões (long DataFrame -> wide),
    calcula retorno esperado por dia/ativo e aplica regras de entrada/saída.

    Filtros opcionais disponíveis:
      - Bandas de Bollinger (bb_window/bb_k) para reforçar tendências.
      - Stop baseado em ATR aproximado (atr_window/atr_stop_k).
      - Cooldown pós-saída e limite máximo de barras por trade.

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

    # 7.1) Bollinger Bands (opcional)
    bb_exit_mask = pd.DataFrame(False, index=exp_ret.index, columns=exp_ret.columns)
    if bb_window is not None and bb_window > 1:
        bb_ma = close_wide.rolling(bb_window, min_periods=bb_window).mean().reindex_like(exp_ret)
        bb_std = close_wide.rolling(bb_window, min_periods=bb_window).std().reindex_like(exp_ret)
        bb_upper = (bb_ma + float(bb_k) * bb_std).reindex_like(exp_ret)
        entries = entries & (close_wide.reindex_like(exp_ret) > bb_upper).fillna(False)
        bb_exit_mask = (close_wide.reindex_like(exp_ret) < bb_ma).fillna(False)

    # 7.2) ATR aproximado (com base em Close) para trailing stop
    atr_frame: Optional[pd.DataFrame] = None
    if atr_window is not None and atr_window > 1 and atr_stop_k is not None and atr_stop_k > 0.0:
        atr_frame = close_wide.diff().abs().rolling(atr_window, min_periods=atr_window).mean().reindex_like(exp_ret)
    else:
        atr_stop_k = None

    # 8) Limpeza fina
    entries = entries.fillna(False)
    exits = exits.fillna(False)

    # 9) Montar dict final, **alinhado ao índice de cada ticker** em close_wide
    signals: Dict[str, Dict[str, pd.Series]] = {}
    needs_iter = (
        only_non_overlapping
        or cooldown_bars > 0
        or (max_hold_bars is not None and max_hold_bars > 0)
        or atr_frame is not None
        or bb_exit_mask.any().any()
    )
    for t in close_wide.columns:
        base_entries = entries.get(t, pd.Series(False, index=exp_ret.index)).reindex(close_wide.index, fill_value=False)
        base_exits = exits.get(t, pd.Series(False, index=exp_ret.index)).reindex(close_wide.index, fill_value=False)

        if needs_iter:
            price = close_wide[t].reindex(close_wide.index)
            bb_exit = bb_exit_mask.get(t, pd.Series(False, index=exp_ret.index)).reindex(close_wide.index, fill_value=False)
            atr_series = (
                atr_frame.get(t, pd.Series(float("nan"), index=exp_ret.index)).reindex(close_wide.index)
                if atr_frame is not None
                else None
            )

            open_pos = False
            entry_price = float("nan")
            hold_bars = 0
            cooldown = 0
            entries_out = []
            exits_out = []

            for ts in price.index:
                price_now = price.loc[ts]

                exit_flag = bool(base_exits.loc[ts]) and open_pos
                if open_pos:
                    hold_bars += 1
                    if max_hold_bars is not None and max_hold_bars > 0 and hold_bars >= max_hold_bars:
                        exit_flag = True
                    if atr_series is not None and atr_stop_k is not None:
                        atr_val = atr_series.loc[ts]
                        if pd.notna(atr_val) and pd.notna(entry_price) and pd.notna(price_now):
                            if price_now <= entry_price - atr_stop_k * atr_val:
                                exit_flag = True
                    if bb_exit.loc[ts]:
                        exit_flag = True
                else:
                    hold_bars = 0

                if exit_flag:
                    exits_out.append(True)
                    open_pos = False
                    entry_price = float("nan")
                    hold_bars = 0
                    cooldown = max(cooldown_bars, 0)
                else:
                    exits_out.append(False)
                    if cooldown > 0:
                        cooldown -= 1

                allow_entry = (not open_pos or not only_non_overlapping) and cooldown == 0
                entry_flag = bool(base_entries.loc[ts]) and allow_entry
                if entry_flag:
                    entries_out.append(True)
                    if not open_pos:
                        entry_price = price_now
                        hold_bars = 0
                    open_pos = True
                else:
                    entries_out.append(False)

            e = pd.Series(entries_out, index=price.index)
            x = pd.Series(exits_out, index=price.index)
        else:
            e = base_entries
            x = base_exits

        signals[t] = {"entries": e.astype(bool), "exits": x.astype(bool)}

        if debug:
            print(f"[signals] {t}: entries={int(e.sum())} exits={int(x.sum())}")

    return signals
