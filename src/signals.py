from __future__ import annotations
import pandas as pd
import numpy as np


def build_signals_from_forecast(
    close_wide: pd.DataFrame, yhat_df: pd.DataFrame, horizon: int = 5, exp_thresh: float = 0.003
) -> dict:
    """
    Constrói sinais (entries/exits) por ticker a partir da previsão de preços.
    Estratégia simples: entra comprado (long) se a variação esperada do D+1
    for > exp_thresh (ex.: 0.3%). Sai após 'horizon' dias (hold-to-horizon).

    Retorna dict[ticker] = {'entries': pd.Series(bool), 'exits': pd.Series(bool)}
    """
    # yhat_df contém h passos; pegamos o 1º passo por data (ds) e ticker
    # Como o NF retorna previsões futuras em 'ds' > último ds observado,
    # vamos alinhar para gerar o sinal no "dia anterior".
    signals = {}
    for ticker in close_wide.columns:
        px = close_wide[ticker].dropna()
        # previsões do ticker
        f = yhat_df[yhat_df["unique_id"] == ticker].set_index("ds").sort_index()
        # estimativa de retorno no próximo dia usando último close disponível
        # alinhamos prev_t = ds_pred; prev_ret_t = y_hat/close_prev - 1
        # precisamos deslocar a previsão para que o sinal seja "na véspera"
        aligned = px.to_frame("close").join(f[["y_hat"]], how="left")
        # a previsão cujo ds é futuro não casa com o índice do px (histórico);
        # então vamos criar sinal baseado na variação esperada entre o último 'close'
        # e a 1ª previsão futura disponível (shift para trás).
        aligned["y_hat_shift1"] = aligned["y_hat"].shift(-1)
        aligned["exp_ret_1d"] = aligned["y_hat_shift1"] / aligned["close"] - 1.0
        entries = (aligned["exp_ret_1d"] > exp_thresh).fillna(False)

        # exit: após 'horizon' dias da entrada
        exits = entries.shift(horizon).fillna(False)

        signals[ticker] = {"entries": entries.astype(bool), "exits": exits.astype(bool)}
    return signals
