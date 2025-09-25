"""Grid search utility to try multiple signal/backtest configurations.

The script reuses the existing pipeline pieces (ingest → prep → NHITS → signals →
backtest) and optionally logs each run into the local registry (SQLite) that we
introduced in ``src/exp_store.py``. It performs early-stop when a result meets
user-defined thresholds.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.config import Cfg, load_config
from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest
from src.exp_store import log_run
from src.run_experiment import _git_hash


def _parse_float_list(values: Optional[Sequence[str]], fallback: float) -> List[Optional[float]]:
    if values is None:
        return [fallback]
    parsed: List[Optional[float]] = []
    for val in values:
        if val.lower() == "none":
            parsed.append(None)
        else:
            parsed.append(float(val))
    return parsed


def _parse_int_list(values: Optional[Sequence[str]], fallback: int) -> List[int]:
    if values is None:
        return [fallback]
    return [int(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for signal/backtest configurations")
    parser.add_argument("--config", type=str, required=True, help="YAML de configuração base")
    parser.add_argument("--lead-for-signal", nargs="+", type=str, default=None, help="Lista de leads (int)")
    parser.add_argument("--dyn-thresh-k", nargs="+", type=str, default=None, help="Lista de k para limiar dinâmico")
    parser.add_argument("--exp-thresh", nargs="+", type=str, default=None, help="Lista de limiares fixos")
    parser.add_argument("--trend-sma", nargs="+", type=str, default=None, help="Lista de SMAs de tendência")
    parser.add_argument("--consec", nargs="+", type=str, default=None, help="Lista de N consecutivos")
    parser.add_argument("--risk-per-trade", nargs="+", type=str, default=None, help="Lista de frações (ex.: 0.005)")
    parser.add_argument("--vol-window", nargs="+", type=str, default=None, help="Lista de janelas de volatilidade")
    parser.add_argument("--rsi-window", nargs="+", type=str, default=None, help="Lista de janelas do RSI")
    parser.add_argument("--rsi-min", nargs="+", type=str, default=None, help="Lista de thresholds mínimo do RSI")
    parser.add_argument("--bb-window", nargs="+", type=str, default=None, help="Lista de janelas das Bandas de Bollinger")
    parser.add_argument("--bb-k", nargs="+", type=str, default=None, help="Lista de multiplicadores das bandas")
    parser.add_argument("--atr-window", nargs="+", type=str, default=None, help="Lista de janelas para ATR aproximado")
    parser.add_argument("--atr-stop-k", nargs="+", type=str, default=None, help="Lista de múltiplos do ATR para stop")
    parser.add_argument("--cooldown-bars", nargs="+", type=str, default=None, help="Lista de períodos de espera pós-saída")
    parser.add_argument("--max-hold-bars", nargs="+", type=str, default=None, help="Lista de limites de barras por trade")

    parser.add_argument("--target-total-return", type=float, default=None, help="Mínimo de retorno médio (%) para early-stop")
    parser.add_argument("--target-sharpe", type=float, default=None, help="Mínimo de Sharpe médio para early-stop")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Máximo de drawdown médio permitido (%)")
    parser.add_argument("--max-combos", type=int, default=None, help="Limite de combinações a avaliar")

    parser.add_argument("--registry-enabled", action="store_true", help="Forçar logging no registry")
    parser.add_argument("--registry-path", type=str, default=None, help="SQLite custom")
    parser.add_argument("--output", type=str, default="reports/grid_search_results.csv")
    return parser.parse_args()


def _load_base_cfg(path: str) -> Cfg:
    cfg = load_config(path)
    return cfg


def _prepare_data(cfg: Cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close_wide = get_prices(cfg.data.tickers, start=cfg.data.start)
    close_wide = close_wide.sort_index()
    prep_out = prepare_long_and_features(close_wide)
    if isinstance(prep_out, pd.DataFrame):
        long_df = prep_out
    elif isinstance(prep_out, dict):
        if "long_df" in prep_out:
            long_df = prep_out["long_df"]
        elif "df_long" in prep_out:
            long_df = prep_out["df_long"]
        else:
            raise KeyError(f"prepare_long_and_features retornou chaves inesperadas: {list(prep_out.keys())}")
    else:
        raise TypeError(f"Tipo inesperado de retorno em prepare_long_and_features: {type(prep_out)}")
    return close_wide, long_df


def _combo_iter(
    leads: List[int],
    dyn_ks: List[Optional[float]],
    exp_thresholds: List[Optional[float]],
    trend_smas: List[Optional[int]],
    consecs: List[int],
    risks: List[Optional[float]],
    vol_windows: List[int],
    rsi_windows: List[Optional[int]],
    rsi_mins: List[Optional[float]],
    bb_windows: List[Optional[int]],
    bb_ks: List[float],
    atr_windows: List[Optional[int]],
    atr_ks: List[Optional[float]],
    cooldowns: List[int],
    max_hold_list: List[Optional[int]],
) -> Iterable[Dict[str, Optional[float]]]:
    for combo in itertools.product(
        leads,
        dyn_ks,
        exp_thresholds,
        trend_smas,
        consecs,
        risks,
        vol_windows,
        rsi_windows,
        rsi_mins,
        bb_windows,
        bb_ks,
        atr_windows,
        atr_ks,
        cooldowns,
        max_hold_list,
    ):
        yield {
            "lead_for_signal": combo[0],
            "dyn_thresh_k": combo[1],
            "exp_thresh": combo[2],
            "trend_sma": combo[3],
            "consec": combo[4],
            "risk_per_trade": combo[5],
            "vol_window": combo[6],
            "rsi_window": combo[7],
            "rsi_min": combo[8],
            "bb_window": combo[9],
            "bb_k": combo[10],
            "atr_window": combo[11],
            "atr_stop_k": combo[12],
            "cooldown_bars": combo[13],
            "max_hold_bars": combo[14],
        }


def meets_targets(
    summary: pd.DataFrame,
    target_total_return: Optional[float],
    target_sharpe: Optional[float],
    max_drawdown: Optional[float],
) -> bool:
    if summary.empty:
        return False
    metrics = summary.mean(numeric_only=True)
    if target_total_return is not None and metrics.get("Total Return [%]", -float("inf")) < target_total_return:
        return False
    if target_sharpe is not None and metrics.get("Sharpe Ratio", -float("inf")) < target_sharpe:
        return False
    if max_drawdown is not None and metrics.get("Max Drawdown [%]", float("inf")) > max_drawdown:
        return False
    return True


def main() -> None:
    args = parse_args()
    base_cfg = _load_base_cfg(args.config)

    close_wide, long_df = _prepare_data(base_cfg)

    leads = _parse_int_list(args.lead_for_signal, base_cfg.model.lead_for_signal)
    dyn_ks = _parse_float_list(args.dyn_thresh_k, base_cfg.signals.dyn_thresh_k or 0.0)
    exp_thresholds = _parse_float_list(args.exp_thresh, base_cfg.signals.exp_thresh)
    trend_smas = _parse_int_list(args.trend_sma, base_cfg.signals.trend_sma or 0)
    consecs = _parse_int_list(args.consec, base_cfg.signals.consec)
    risks = _parse_float_list(args.risk_per_trade, base_cfg.backtest.risk_per_trade or 0.0)
    vol_windows = _parse_int_list(args.vol_window, base_cfg.signals.vol_window)
    rsi_windows = _parse_int_list(args.rsi_window, base_cfg.signals.rsi_window or 0)
    rsi_mins = _parse_float_list(args.rsi_min, base_cfg.signals.rsi_min or 0.0)
    bb_windows = _parse_int_list(args.bb_window, base_cfg.signals.bb_window or 0)
    bb_ks = _parse_float_list(args.bb_k, base_cfg.signals.bb_k or 2.0)
    atr_windows = _parse_int_list(args.atr_window, base_cfg.signals.atr_window or 0)
    atr_ks = _parse_float_list(args.atr_stop_k, base_cfg.signals.atr_stop_k or 0.0)
    cooldowns = _parse_int_list(args.cooldown_bars, base_cfg.signals.cooldown_bars)
    max_hold_list = _parse_int_list(args.max_hold_bars, base_cfg.signals.max_hold_bars or 0)

    # Ensure optional integers are handled (0 -> None)
    trend_smas = [None if val == 0 else val for val in trend_smas]
    rsi_windows = [None if val == 0 else val for val in rsi_windows]
    rsi_mins = [None if val == 0.0 else val for val in rsi_mins]
    dyn_ks = [None if val == 0.0 else val for val in dyn_ks]
    risks = [None if val == 0.0 else val for val in risks]
    bb_windows = [None if val == 0 else val for val in bb_windows]
    atr_windows = [None if val == 0 else val for val in atr_windows]
    atr_ks = [None if (val is None or val == 0.0) else val for val in atr_ks]
    max_hold_list = [None if val == 0 else val for val in max_hold_list]

    combos = _combo_iter(
        leads,
        dyn_ks,
        exp_thresholds,
        trend_smas,
        consecs,
        risks,
        vol_windows,
        rsi_windows,
        rsi_mins,
        bb_windows,
        [float(v) for v in bb_ks],
        atr_windows,
        atr_ks,
        cooldowns,
        max_hold_list,
    )

    results: List[Dict[str, Optional[float]]] = []
    git_hash = _git_hash()
    registry_enabled = args.registry_enabled or base_cfg.registry.enabled
    registry_path = args.registry_path or base_cfg.registry.path

    yhat_cache: Dict[int, pd.DataFrame] = {}

    for idx, combo in enumerate(combos, start=1):
        if args.max_combos is not None and idx > args.max_combos:
            print("[grid] Limite de combinações atingido; encerrando.")
            break

        lead = combo["lead_for_signal"]
        if lead not in yhat_cache:
            cfg_model = base_cfg.model.model_copy(deep=True)
            cfg_model.lead_for_signal = lead
            print(f"[grid] Treinando modelo para lead={lead} (combo #{idx})")
            yhat_cache[lead] = train_predict_nhits(
                long_df=long_df,
                horizon=cfg_model.horizon,
                input_size=cfg_model.input_size,
                n_windows=cfg_model.n_windows,
                step_size=cfg_model.step_size,
                max_steps=cfg_model.max_steps,
                seed=cfg_model.seed,
                lead_for_signal=cfg_model.lead_for_signal,
            )
        yhat_df = yhat_cache[lead]

        cfg = base_cfg.model_copy(deep=True)
        cfg.model.lead_for_signal = lead
        cfg.signals.dyn_thresh_k = combo["dyn_thresh_k"]
        cfg.signals.exp_thresh = combo["exp_thresh"] if combo["exp_thresh"] is not None else cfg.signals.exp_thresh
        cfg.signals.trend_sma = combo["trend_sma"]
        cfg.signals.consec = combo["consec"]
        cfg.signals.vol_window = combo["vol_window"]
        cfg.signals.rsi_window = combo["rsi_window"]
        cfg.signals.rsi_min = combo["rsi_min"]
        cfg.signals.bb_window = combo["bb_window"]
        cfg.signals.bb_k = combo["bb_k"]
        cfg.signals.atr_window = combo["atr_window"]
        cfg.signals.atr_stop_k = combo["atr_stop_k"]
        cfg.signals.cooldown_bars = combo["cooldown_bars"]
        cfg.signals.max_hold_bars = combo["max_hold_bars"]
        cfg.backtest.risk_per_trade = combo["risk_per_trade"]

        print(
            "[grid] Rodando sinais/backtest: "
            f"lead={lead} dyn_k={cfg.signals.dyn_thresh_k} exp_thresh={cfg.signals.exp_thresh} "
            f"trend_sma={cfg.signals.trend_sma} consec={cfg.signals.consec} "
            f"risk_per_trade={cfg.backtest.risk_per_trade} vol_window={cfg.signals.vol_window} "
            f"rsi_window={cfg.signals.rsi_window} rsi_min={cfg.signals.rsi_min} "
            f"bb_window={cfg.signals.bb_window} bb_k={cfg.signals.bb_k} "
            f"atr_window={cfg.signals.atr_window} atr_stop_k={cfg.signals.atr_stop_k} "
            f"cooldown={cfg.signals.cooldown_bars} max_hold={cfg.signals.max_hold_bars}"
        )

        signals = build_signals_from_forecast(
            forecast_df=yhat_df,
            close_wide=close_wide,
            exp_thresh=cfg.signals.exp_thresh,
            consec=cfg.signals.consec,
            trend_sma=cfg.signals.trend_sma,
            dyn_thresh_k=cfg.signals.dyn_thresh_k,
            vol_window=cfg.signals.vol_window,
            rsi_window=cfg.signals.rsi_window,
            rsi_min=cfg.signals.rsi_min,
            bb_window=cfg.signals.bb_window,
            bb_k=cfg.signals.bb_k,
            atr_window=cfg.signals.atr_window,
            atr_stop_k=cfg.signals.atr_stop_k,
            cooldown_bars=cfg.signals.cooldown_bars,
            max_hold_bars=cfg.signals.max_hold_bars,
            only_non_overlapping=cfg.backtest.only_non_overlapping,
            debug=False,
        )

        size_wide = None
        if cfg.backtest.risk_per_trade:
            vol = close_wide.pct_change().rolling(cfg.signals.vol_window, min_periods=cfg.signals.vol_window).std()
            vol = vol.reindex(close_wide.index).fillna(method="ffill").replace(0.0, 1e-4)
            risk_cash = cfg.backtest.risk_per_trade * cfg.backtest.init_cash
            size_wide = risk_cash / (vol * close_wide)
            size_wide = size_wide.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
            size_wide = size_wide.clip(lower=0.0).applymap(lambda x: float(max(0, int(x))))

        summary_df, _ = run_backtest(
            close_wide=close_wide,
            signals=signals,
            init_cash=cfg.backtest.init_cash,
            fees=cfg.backtest.fees,
            slippage=cfg.backtest.slippage,
            direction=cfg.backtest.direction,
            save_trades=False,
            report_path=Path("reports") / f"summary_grid_{idx:04d}.csv",
            size_wide=size_wide,
        )

        run_id = None
        if registry_enabled:
            try:
                run_id = log_run(
                    db_path=registry_path,
                    cfg=cfg,
                    summary_df=summary_df,
                    report_path=Path("reports") / f"summary_grid_{idx:04d}.csv",
                    git_hash=git_hash,
                )
                print(f"[grid] Run {run_id} salvo no registry")
            except Exception as exc:
                print(f"[grid] Falha ao registrar run: {exc}")

        metrics_mean = summary_df.mean(numeric_only=True)
        result_entry = {
            "combo_index": idx,
            "run_id": run_id,
            "lead_for_signal": lead,
            "dyn_thresh_k": cfg.signals.dyn_thresh_k,
            "exp_thresh": cfg.signals.exp_thresh,
            "trend_sma": cfg.signals.trend_sma,
            "consec": cfg.signals.consec,
            "risk_per_trade": cfg.backtest.risk_per_trade,
            "vol_window": cfg.signals.vol_window,
            "rsi_window": cfg.signals.rsi_window,
            "rsi_min": cfg.signals.rsi_min,
            "bb_window": cfg.signals.bb_window,
            "bb_k": cfg.signals.bb_k,
            "atr_window": cfg.signals.atr_window,
            "atr_stop_k": cfg.signals.atr_stop_k,
            "cooldown_bars": cfg.signals.cooldown_bars,
            "max_hold_bars": cfg.signals.max_hold_bars,
            "avg_total_return": float(metrics_mean.get("Total Return [%]", float("nan"))),
            "avg_sharpe": float(metrics_mean.get("Sharpe Ratio", float("nan"))),
            "avg_max_drawdown": float(metrics_mean.get("Max Drawdown [%]", float("nan"))),
            "avg_win_rate": float(metrics_mean.get("Win Rate [%]", float("nan"))),
            "total_trades": int(summary_df["Trades"].sum()) if "Trades" in summary_df else 0,
        }
        results.append(result_entry)

        if meets_targets(summary_df, args.target_total_return, args.target_sharpe, args.max_drawdown):
            print("[grid] Critérios atingidos; encerrando busca antecipadamente.")
            break

    if results:
        df_results = pd.DataFrame(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"[grid] Resultados salvos em {output_path}")
    else:
        print("[grid] Nenhuma combinação foi avaliada.")


if __name__ == "__main__":
    main()
