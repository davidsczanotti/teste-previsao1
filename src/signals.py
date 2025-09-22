# src/signals.py
from __future__ import annotations

from typing import Dict, Union, Optional
import numpy as np
import pandas as pd


Number = Union[int, float]
MaybeDictF = Union[Number, Dict[str, Number]]
MaybeDictI = Union[int, Dict[str, int]]


def _get_param_per_ticker(param: Union[Number, Dict[str, Number]], ticker: str, default: Number) -> Number:
    """Devolve param[ticker] se for dict, senão o valor escalar."""
    if isinstance(param, dict):
        return param.get(ticker, default)
    return param


def _ensure_unique_id(yhat_df: pd.DataFrame, fallback_tickers: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Garante colunas: ['unique_id','ds','y_hat'].
    - Se não houver 'unique_id', tenta inferir: se só há 1 ticker, preenche com esse ticker.
    - Converte 'ds' para datetime e ordena.
    """
    cols = {c.lower(): c for c in yhat_df.columns}
    # normalizar nomes de colunas
    uid_col = cols.get("unique_id", None)
    ds_col = cols.get("ds", None)
    yhat_col = cols.get("y_hat", None)
    if yhat_col is None:
        raise ValueError("yhat_df precisa ter a coluna 'y_hat'.")

    df = yhat_df.copy()
    if ds_col is None:
        raise ValueError("yhat_df precisa ter a coluna 'ds' (datas previstas).")
    df.rename(columns={yhat_col: "y_hat", ds_col: "ds"}, inplace=True)

    if uid_col is None:
        # tentativa: se só vier de 1 ativo, criamos essa coluna
        if fallback_tickers is None or len(fallback_tickers) != 1:
            # se não dá pra inferir, tudo como 'UNKNOWN'
            df["unique_id"] = "UNKNOWN"
        else:
            df["unique_id"] = fallback_tickers[0]
    else:
        df.rename(columns={uid_col: "unique_id"}, inplace=True)

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df[["unique_id", "ds", "y_hat"]]


def _series_from_yhat_for_ticker(yhat_df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Para um ticker, devolve a série y_hat indexada por ds (ordenada).
    """
    f = yhat_df[yhat_df["unique_id"] == ticker].copy()
    if f.empty:
        # retorna série vazia com dtype float
        return pd.Series(dtype=float, name="y_hat")
    f = f.sort_values("ds")
    s = f.set_index("ds")["y_hat"].astype(float).copy()
    s.name = "y_hat"
    return s


def _rolling_vol20(close: pd.Series) -> pd.Series:
    """Volatilidade (desvio padrão) de 20 dias em retornos diários (não anualizada)."""
    return close.pct_change().rolling(20, min_periods=20).std()


def _sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def _generate_entries_exits(
    exp_hat: pd.Series,
    thr: Union[Number, pd.Series],
    trend_ok: Optional[pd.Series],
    min_hold: int = 1,
    early_exit_on_flip: bool = True,
    exit_symmetric: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Constrói entries/exits walk-forward respeitando min_hold, saída antecipada e saída simétrica.
    exp_hat, thr e trend_ok devem estar alinhados ao mesmo índice temporal.
    """
    # alinhar
    if isinstance(thr, (int, float)):
        thr = pd.Series(thr, index=exp_hat.index)
    else:
        thr = thr.reindex(exp_hat.index)

    if trend_ok is None:
        trend_ok = pd.Series(True, index=exp_hat.index)
    else:
        trend_ok = trend_ok.reindex(exp_hat.index).fillna(False)

    # sinal "ligado" (condição de entrada)
    long_on = (exp_hat > thr) & trend_ok

    entries = pd.Series(False, index=exp_hat.index)
    exits = pd.Series(False, index=exp_hat.index)

    in_pos = False
    hold = 0

    for i, ts in enumerate(exp_hat.index):
        if not in_pos:
            # entrar somente se habilitado
            if bool(long_on.iloc[i]):
                entries.iloc[i] = True
                in_pos = True
                hold = 1
            continue

        # se já está posicionado:
        hold += 1

        # regra de saída
        do_exit = False

        if exit_symmetric:
            # sai quando exp_hat < -thr
            if exp_hat.iloc[i] < -thr.iloc[i]:
                do_exit = True
        else:
            # padrão: sai quando exp_hat < 0
            if exp_hat.iloc[i] < 0:
                do_exit = True
            # saída antecipada: se "flipar" (long_on falso) e permitir sair antes
            if early_exit_on_flip and (not long_on.iloc[i]):
                do_exit = True

        # respeitar min_hold
        if do_exit and hold >= max(1, int(min_hold)):
            exits.iloc[i] = True
            in_pos = False
            hold = 0

    return entries.astype(bool), exits.astype(bool)


def build_signals_from_forecast(
    close_wide: pd.DataFrame,
    yhat_df: pd.DataFrame,
    horizon: MaybeDictI = 5,
    exp_thresh: MaybeDictF = 0.003,
    *,
    use_vol_threshold: bool = False,
    vol_k: MaybeDictF = 0.0,
    trend_sma: Optional[Union[int, Dict[str, int]]] = None,
    early_exit_on_flip: bool = True,
    exit_symmetric: bool = False,
    min_hold: MaybeDictI = 1,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Constrói sinais (entries/exits) por ticker usando previsões y_hat.

    Parâmetros principais
    ---------------------
    close_wide : DataFrame [Date x Ticker] com preços de fechamento.
    yhat_df    : DataFrame com colunas ['unique_id','ds','y_hat'] (ou equivalente).
    horizon    : int ou dict por ticker (nº de dias à frente).
    exp_thresh : limiar fixo de ganho esperado (se use_vol_threshold=False).
    use_vol_threshold : se True, o limiar vira vol_k[t] * vol20[t].
    vol_k      : float ou dict por ticker (multiplicador de vol).
    trend_sma  : None, int (mesmo para todos) ou dict por ticker (filtro de tendência).
    early_exit_on_flip : sai quando o sinal "desliga".
    exit_symmetric     : se True, saída quando exp_hat < -threshold.
    min_hold   : int ou dict por ticker (mínimo de dias em posição).

    Retorno
    -------
    signals : dict[ticker] -> {"entries": Series[bool], "exits": Series[bool]}
              Alinhado ao índice de close_wide.
    """
    assert isinstance(close_wide, pd.DataFrame), "close_wide deve ser um DataFrame wide (Date x Ticker)."
    tickers = list(close_wide.columns)
    yhat_df = _ensure_unique_id(yhat_df, fallback_tickers=tickers)

    signals: Dict[str, Dict[str, pd.Series]] = {}

    for ticker in tickers:
        c = close_wide[ticker].dropna()
        if c.empty:
            # nada a fazer
            signals[ticker] = {
                "entries": pd.Series(False, index=close_wide.index),
                "exits": pd.Series(False, index=close_wide.index),
            }
            print(f"DEBUG {ticker}: entries=0 exits=0")
            continue

        # horizonte por ativo
        if isinstance(horizon, dict):
            h_t = int(horizon.get(ticker, max(horizon.values())))
        else:
            h_t = int(horizon)
        h_t = max(1, h_t)

        # série de previsões (indexada por 'ds')
        y_series = _series_from_yhat_for_ticker(yhat_df, ticker)

        # alinhar ao mesmo índice de preços
        # (assumindo que 'ds' são dias de negociação compatíveis com o índice de close)
        if not y_series.index.is_unique:
            y_series = y_series[~y_series.index.duplicated(keep="last")]

        y_series = y_series.reindex(close_wide.index).copy() 

        # IMPORTANTÍSSIMO:
        # usar o passo do horizonte -> deslocar para trás h_t dias: em t, avaliamos y_hat de t+h_t
        # exp_hat[t] = y_hat[t+h_t] / close[t] - 1
        y_target = y_series.shift(-h_t)
        exp_hat = (y_target / c) - 1.0
        exp_hat = exp_hat.reindex(close_wide.index)

        # limiar (threshold)
        if use_vol_threshold:
            vol20 = _rolling_vol20(c)
            k = _get_param_per_ticker(vol_k, ticker, 0.0)
            thr = k * vol20
        else:
            thr_val = _get_param_per_ticker(exp_thresh, ticker, 0.0)
            thr = pd.Series(thr_val, index=close_wide.index)

        # filtro de tendência (opcional)
        if trend_sma is None:
            trend_ok = pd.Series(True, index=close_wide.index)
        else:
            if isinstance(trend_sma, dict):
                w = int(trend_sma.get(ticker, 0))
            else:
                w = int(trend_sma)
            if w > 0:
                sma = _sma(c, w)
                trend_ok = (c > sma).reindex(close_wide.index).fillna(False)
            else:
                trend_ok = pd.Series(True, index=close_wide.index)

        # min_hold por ticker
        mh = _get_param_per_ticker(min_hold, ticker, 1)
        mh = int(max(1, mh))

        entries, exits = _generate_entries_exits(
            exp_hat=exp_hat.fillna(0.0),
            thr=thr.fillna(np.inf if use_vol_threshold else 0.0),
            trend_ok=trend_ok,
            min_hold=mh,
            early_exit_on_flip=early_exit_on_flip,
            exit_symmetric=exit_symmetric,
        )

        # salvar
        signals[ticker] = {"entries": entries, "exits": exits}

        # debug similar ao que você já vê
        print(
            f"DEBUG {ticker}: entries={int(entries.sum())} exits={int(exits.sum())}"
        )

    return signals
