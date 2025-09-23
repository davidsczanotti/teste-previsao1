from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _dedup_index_keep_last(s: pd.Series) -> pd.Series:
    """
    Remove duplicatas de índice mantendo o último valor.
    Útil quando há previsões 'rolling' para a mesma data.
    """
    if s.index.has_duplicates:
        # agrupa por timestamp e mantém o último valor reportado
        s = s.groupby(level=0).last()
    return s


def _as_bool_series(x: pd.Series, index: pd.Index) -> pd.Series:
    """Garante série booleana alinhada ao índice alvo."""
    if x is None:
        out = pd.Series(False, index=index)
    else:
        out = x.reindex(index, fill_value=False)
    if out.dtype != bool:
        out = out.astype(bool)
    return out


def build_signals_from_forecast(
    close_wide: pd.DataFrame,
    yhat_df: pd.DataFrame,
    horizon: int = 5,
    exp_thresh: float = 0.003,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Constrói sinais de entrada/saída por ticker a partir das previsões (yhat_df).

    Estratégia simples (baseline):
      - Para cada data t, com preço atual P_t e previsão y_hat_t (próximo candle/curto prazo),
        calcula-se retorno esperado R_exp = y_hat_t / P_t - 1.
      - entries = R_exp > +exp_thresh
      - exits   = R_exp < -exp_thresh

    Observações importantes:
      • NÃO aplicamos execução aqui. O no-lookahead é garantido depois, no backtest,
        com shift(1) nas séries de entries/exits.
      • Tratamos duplicatas de 'ds' no yhat_df (mantemos o último valor por data).
      • Alinhamos tudo no índice de preços do ticker para evitar reindex com duplicatas.

    Parâmetros
    ----------
    close_wide : DataFrame
        Fechamentos, colunas = tickers, índice = datas.
    yhat_df : DataFrame
        Colunas esperadas: ['unique_id', 'ds', 'y_hat'].
    horizon : int
        Horizonte de previsão (não usado explicitamente aqui, mas mantido para compatibilidade).
    exp_thresh : float
        Limiar absoluto de retorno esperado para acionar sinal.

    Retorna
    -------
    dict
        {ticker: {"entries": Series[bool], "exits": Series[bool]}}
    """
    if not {"unique_id", "ds", "y_hat"}.issubset(yhat_df.columns):
        raise ValueError("yhat_df deve conter as colunas: 'unique_id', 'ds', 'y_hat'.")

    # garantir tipos
    yhat_df = yhat_df.copy()
    yhat_df["ds"] = pd.to_datetime(yhat_df["ds"])
    yhat_df = yhat_df.sort_values(["unique_id", "ds"])

    signals: Dict[str, Dict[str, pd.Series]] = {}

    for ticker in close_wide.columns:
        # 1) Série de preços desse ticker
        px = close_wide[ticker].dropna()
        if px.empty:
            continue

        # 2) Previsões desse ticker
        f_tk = yhat_df[yhat_df["unique_id"] == ticker]
        if f_tk.empty:
            # sem previsão → sem sinal
            signals[ticker] = {
                "entries": pd.Series(False, index=px.index),
                "exits": pd.Series(False, index=px.index),
            }
            continue

        # 3) série prevista y_hat por data
        y_series = (
            f_tk[["ds", "y_hat"]]
            .dropna()
            .drop_duplicates(subset=["ds"], keep="last")  # se vier duplicado
            .set_index("ds")["y_hat"]
        )

        # 4) se ainda houver duplicata de índice por algum motivo, consolidar
        y_series = _dedup_index_keep_last(y_series)

        # 5) alinhar ao índice de preços (sem duplicatas)
        if y_series.index.has_duplicates:
            # segurança extra (em teoria, já tratamos)
            y_series = y_series[~y_series.index.duplicated(keep="last")]

        # agora podemos reindexar; prever para datas sem previsão vira NaN
        y_series = y_series.reindex(px.index)

        # 6) retorno esperado
        #    Obs: se y_hat vier como preço “no mesmo timestamp t”, o backtest fará shift(1)
        #    para executar no próximo candle.
        with np.errstate(divide="ignore", invalid="ignore"):
            r_exp = y_series / px - 1.0

        # 7) sinais (booleans) com limiar
        entries = (r_exp > +exp_thresh).fillna(False)
        exits = (r_exp < -exp_thresh).fillna(False)

        # 8) garantir dtype/índice correto
        entries = _as_bool_series(entries, px.index)
        exits = _as_bool_series(exits, px.index)

        # 9) sem prints aqui (debug centralizado no backtest)
        signals[ticker] = {"entries": entries, "exits": exits}

    return signals
