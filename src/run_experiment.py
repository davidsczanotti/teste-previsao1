# src/run_experiment.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess
from typing import Optional, Dict

import pandas as pd

from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest
from src.config import load_config, Cfg

# MLflow é opcional
try:
    import mlflow
except Exception:
    mlflow = None


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML de configuração")
    # CLI “curta” mantém compat se alguém quiser rodar sem YAML
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--n-windows", type=int, default=None)
    p.add_argument("--step-size", type=int, default=1)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exp-thresh", type=float, default=None)
    p.add_argument("--consec", type=int, default=None)
    p.add_argument("--trend-sma", type=int, default=None)
    p.add_argument("--dyn-thresh-k", type=float, default=None)
    p.add_argument("--vol-window", type=int, default=None)
    # RSI opcional
    p.add_argument("--rsi-window", type=int, default=None)
    p.add_argument("--rsi-min", type=float, default=None)
    p.add_argument("--fees", type=float, default=None)
    p.add_argument("--slippage", type=float, default=None)
    p.add_argument("--init-cash", type=float, default=None)
    return p.parse_args(argv)


def merge_cli_over_config(cfg: Cfg, args: argparse.Namespace) -> Cfg:
    # data
    if args.tickers:
        cfg.data.tickers = args.tickers
    if args.start:
        cfg.data.start = args.start
    # model
    if args.horizon is not None:
        cfg.model.horizon = args.horizon
    if args.n_windows is not None:
        cfg.model.n_windows = args.n_windows
    if args.step_size is not None:
        cfg.model.step_size = args.step_size
    if args.input_size is not None:
        cfg.model.input_size = args.input_size
    if args.max_steps is not None:
        cfg.model.max_steps = args.max_steps
    # signals
    if args.exp_thresh is not None:
        cfg.signals.exp_thresh = args.exp_thresh
    if args.consec is not None:
        cfg.signals.consec = args.consec
    if args.trend_sma is not None:
        cfg.signals.trend_sma = args.trend_sma
    if args.dyn_thresh_k is not None:
        cfg.signals.dyn_thresh_k = args.dyn_thresh_k
    if args.vol_window is not None:
        cfg.signals.vol_window = args.vol_window
    if args.rsi_window is not None:
        cfg.signals.rsi_window = args.rsi_window
    if args.rsi_min is not None:
        cfg.signals.rsi_min = args.rsi_min
    # backtest
    if args.fees is not None:
        cfg.backtest.fees = args.fees
    if args.slippage is not None:
        cfg.backtest.slippage = args.slippage
    if args.init_cash is not None:
        cfg.backtest.init_cash = args.init_cash
    return cfg


def maybe_start_mlflow(cfg: Cfg):
    if not cfg.tracking.use_mlflow or mlflow is None:
        return None
    if cfg.tracking.mlflow_uri:
        mlflow.set_tracking_uri(cfg.tracking.mlflow_uri)
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
    run = mlflow.start_run(run_name=cfg.experiment.name)
    # log params base
    mlflow.log_params(
        {
            "tickers": ",".join(cfg.data.tickers),
            "start": cfg.data.start,
            **cfg.model.dict(),
            **cfg.signals.dict(),
            **cfg.backtest.dict(),
            "git_hash": _git_hash() or "unknown",
            "notes": cfg.experiment.notes or "",
        }
    )
    return run


def main(argv=None):
    args = parse_args(argv)

    # 1) Carregar config (YAML + overrides da CLI)
    if args.config:
        cfg = load_config(args.config)
    else:
        # fallback mínimo se não passar YAML
        cfg = Cfg.parse_obj(
            {
                "data": {"tickers": args.tickers or ["VALE3.SA", "PETR4.SA"], "start": args.start or "2020-01-01"},
                "model": {
                    "horizon": args.horizon or 5,
                    "input_size": args.input_size or 60,
                    "n_windows": args.n_windows or 8,
                    "step_size": args.step_size or 1,
                    "max_steps": args.max_steps or 300,
                    "seed": 1,
                    "lead_for_signal": 1,
                },
                "signals": {
                    "exp_thresh": args.exp_thresh or 0.002,
                    "consec": args.consec or 1,
                    "trend_sma": args.trend_sma,
                    "dyn_thresh_k": args.dyn_thresh_k,
                    "vol_window": args.vol_window or 20,
                    "rsi_window": args.rsi_window,
                    "rsi_min": args.rsi_min,
                },
                "backtest": {
                    "init_cash": args.init_cash or 100000,
                    "fees": args.fees or 0.0005,
                    "slippage": args.slippage or 0.0005,
                    "direction": "longonly",
                    "only_non_overlapping": True,
                    "risk_per_trade": None,
                },
                "tracking": {"use_mlflow": False, "mlflow_experiment": "default", "mlflow_uri": None},
                "experiment": {"name": "run", "notes": None},
            }
        )
    cfg = merge_cli_over_config(cfg, args)

    # 2) Dados
    print("1) Baixando dados...")
    close_wide: pd.DataFrame = get_prices(tickers=cfg.data.tickers, start=cfg.data.start)
    close_wide = close_wide.sort_index()
    print(close_wide.head())

    print("2) Preparando long_df + features básicas...")
    prep_out = prepare_long_and_features(close_wide)

    # Aceita tanto dict {"long_df": df} quanto df direto
    if isinstance(prep_out, pd.DataFrame):
        long_df = prep_out
    elif isinstance(prep_out, dict):
        if "long_df" in prep_out:
            long_df = prep_out["long_df"]
        elif "df_long" in prep_out:  # compat antigo
            long_df = prep_out["df_long"]
        else:
            raise KeyError(
                "prepare_long_and_features() deve retornar DataFrame ou dict com 'long_df'/'df_long'. "
                f"Chaves encontradas: {list(prep_out.keys())}"
            )
    else:
        raise TypeError("prepare_long_and_features() retornou tipo inesperado. " f"Tipo: {type(prep_out)}")

    # 3) Modelo -> previsões alinhadas na decisão (cutoff)
    print("3) Treinando NHITS (rolling) e prevendo h=5...")
    yhat_df: pd.DataFrame = train_predict_nhits(
        long_df=long_df,
        horizon=cfg.model.horizon,
        input_size=cfg.model.input_size,
        n_windows=cfg.model.n_windows,
        step_size=cfg.model.step_size,
        max_steps=cfg.model.max_steps,
        seed=cfg.model.seed,
        lead_for_signal=cfg.model.lead_for_signal,
    )

    print("4) Gerando sinais a partir das previsões...")
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
        only_non_overlapping=cfg.backtest.only_non_overlapping,
        debug=True,
    )

    # (Opcional) tamanho de posição inverso à vol: aqui fica simples (estático por barra)
    size_wide = None
    if cfg.backtest.risk_per_trade:
        # shares ~ (risk_per_trade * init_cash) / (vol * price)
        vol = close_wide.pct_change().rolling(cfg.signals.vol_window, min_periods=cfg.signals.vol_window).std()
        vol = vol.reindex(close_wide.index).fillna(method="ffill")
        vol = vol.replace(0.0, 1e-4)
        risk_cash = cfg.backtest.risk_per_trade * cfg.backtest.init_cash
        size_wide = (risk_cash / (vol * close_wide)).clip(lower=0.0)
        # arredonda para inteiro de shares (conservador)
        size_wide = size_wide.applymap(lambda x: float(max(0, int(x))))

    # 5) Backtest
    print("5) Backtest por ticker (vectorbt)...")
    summary_df, pf_dict = run_backtest(
        close_wide=close_wide,
        signals=signals,
        init_cash=cfg.backtest.init_cash,
        fees=cfg.backtest.fees,
        slippage=cfg.backtest.slippage,
        direction=cfg.backtest.direction,
        save_trades=True,
        report_path="reports/summary_baseline.csv",
        size_wide=size_wide,
    )

    print("\n==== RESUMO ====\n")
    print(summary_df)
    print("\nRelatório salvo em: reports/summary_baseline.csv")

    # MLflow
    run = maybe_start_mlflow(cfg)
    if run is not None:
        try:
            # log metrics resumidas
            for k, v in summary_df.mean().to_dict().items():
                mlflow.log_metric(f"mean_{k.replace(' ', '_')}", float(v))
            # artefatos
            mlflow.log_artifact("reports/summary_baseline.csv")
            # trades individuais
            for t in cfg.data.tickers:
                p = Path("reports") / f"trades_{t}.csv"
                if p.exists():
                    mlflow.log_artifact(str(p))
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    main(sys.argv[1:])
