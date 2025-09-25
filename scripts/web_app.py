from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

from src.ingest import get_prices
from src.prep import prepare_long_and_features
from src.models_ts import train_predict_nhits
from src.signals import build_signals_from_forecast
from src.backtest import run_backtest
from src.config import Cfg
from src.exp_store import log_run, last_runs


_ROOT = Path(__file__).resolve().parents[1]
# Garantir que o diretório raiz esteja no sys.path para permitir `import src.*`
root_str = str(_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
APP = Flask(
    __name__,
    template_folder=str(_ROOT / "templates"),
    static_folder=None,
)
APP.secret_key = "dev-secret"


def _load_universe(path: str = "configs/universe_b3.txt") -> List[str]:
    p = Path(path)
    if not p.exists():
        return ["VALE3.SA", "PETR4.SA", "BOVA11.SA", "ITUB4.SA"]
    out: List[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def run_pipeline_for_ticker(
    ticker: str,
    aporte: float,
    start: str,
    profile: str = "B",  # A ou B
) -> dict:
    # Config de perfil
    lead = 2
    trend_sma = 100
    dyn_k = 0.8
    exp_thresh = 0.0
    atr_window = 14
    atr_k = 1.5
    risk = 0.005 if profile.upper() == "B" else None

    close = get_prices([ticker], start=start).sort_index()
    prep_out = prepare_long_and_features(close)
    long_df = prep_out if isinstance(prep_out, pd.DataFrame) else prep_out["long_df"]

    yhat = train_predict_nhits(
        long_df=long_df,
        horizon=5,
        input_size=60,
        n_windows=24,
        step_size=5,
        max_steps=50,
        seed=1,
        lead_for_signal=lead,
    )

    signals = build_signals_from_forecast(
        forecast_df=yhat,
        close_wide=close,
        exp_thresh=exp_thresh,
        consec=1,
        trend_sma=trend_sma,
        dyn_thresh_k=dyn_k,
        vol_window=20,
        atr_window=atr_window,
        atr_stop_k=atr_k,
        cooldown_bars=0,
        only_non_overlapping=True,
        debug=False,
    )

    size_wide = None
    if risk:
        vol = (
            close.pct_change()
            .rolling(20, min_periods=20)
            .std()
            .reindex(close.index)
            .fillna(method="ffill")
            .replace(0.0, 1e-4)
        )
        risk_cash = risk * aporte
        size_wide = (risk_cash / (vol * close)).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0).clip(lower=0.0)
        size_wide = size_wide.applymap(lambda x: float(max(0, int(x))))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path("reports") / f"web_summary_{ticker}_{ts}.csv"

    summary, pf = run_backtest(
        close_wide=close,
        signals=signals,
        init_cash=aporte,
        fees=0.0005,
        slippage=0.0005,
        direction="longonly",
        save_trades=True,
        report_path=report_path,
        size_wide=size_wide,
        aggregate_portfolio=False,
    )

    # Copiar trades para arquivo único por execução
    trades_src = Path("reports") / f"trades_{ticker}.csv"
    trades_dst = Path("reports") / f"web_trades_{ticker}_{ts}.csv"
    if trades_src.exists():
        trades_dst.write_text(trades_src.read_text())

    # Registrar no registry
    cfg = Cfg.parse_obj(
        {
            "data": {"tickers": [ticker], "start": start},
            "model": {
                "horizon": 5,
                "input_size": 60,
                "n_windows": 24,
                "step_size": 5,
                "max_steps": 50,
                "seed": 1,
                "lead_for_signal": lead,
            },
            "signals": {
                "exp_thresh": exp_thresh,
                "consec": 1,
                "trend_sma": trend_sma,
                "dyn_thresh_k": dyn_k,
                "vol_window": 20,
                "atr_window": atr_window,
                "atr_stop_k": atr_k,
                "cooldown_bars": 0,
                "max_hold_bars": None,
            },
            "backtest": {
                "init_cash": aporte,
                "fees": 0.0005,
                "slippage": 0.0005,
                "direction": "longonly",
                "only_non_overlapping": True,
                "risk_per_trade": risk,
            },
            "tracking": {"use_mlflow": False, "mlflow_experiment": "web", "mlflow_uri": None},
            "registry": {"enabled": True, "path": "reports/experiments.sqlite"},
            "experiment": {"name": f"web_{ticker}", "notes": profile},
        }
    )
    try:
        log_run("reports/experiments.sqlite", cfg=cfg, summary_df=summary, report_path=report_path)
    except Exception:
        pass

    rec = summary.reset_index().to_dict(orient="records")
    row = next((r for r in rec if r.get("ticker") == ticker), rec[0] if rec else {})
    return {
        "ticker": ticker,
        "summary_path": str(report_path),
        "trades_path": str(trades_dst) if trades_dst.exists() else None,
        "row": row,
        "ts": ts,
    }


@APP.route("/", methods=["GET", "POST"])
def index():
    tickers = _load_universe()
    result = None
    if request.method == "POST":
        ticker = request.form.get("ticker")
        aporte = float(request.form.get("aporte", "100000") or 100000)
        start = request.form.get("start") or "2018-01-01"
        profile = request.form.get("profile", "B")
        try:
            result = run_pipeline_for_ticker(ticker, aporte, start, profile)
            flash(f"Execução concluída para {ticker}", "success")
        except Exception as exc:
            flash(f"Falha na execução: {exc}", "error")
    # últimas execuções do registry
    try:
        recent = last_runs("reports/experiments.sqlite", limit=15)
    except Exception:
        recent = pd.DataFrame()
    return render_template("index.html", tickers=tickers, result=result, recent=recent)


def create_app() -> Flask:
    return APP


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=True)
