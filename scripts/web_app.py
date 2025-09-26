from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
from collections import OrderedDict
from typing import List, Any, Optional

from flask import Flask, render_template, request, flash
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


# guarda resultados recentes por ticker para exibição no painel
LATEST_RESULTS: "OrderedDict[str, dict]" = OrderedDict()
MAX_RESULTS = 8


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


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return float(value)
        out = float(value)
        return None if pd.isna(out) else out
    except Exception:
        return None


def _series_from_pf(pf) -> pd.Series | None:
    for attr in ("value", "portfolio_value"):
        try:
            candidate = getattr(pf, attr, None)
            if candidate is None:
                continue
            series = candidate() if callable(candidate) else candidate
            if isinstance(series, pd.Series) and not series.empty:
                return series
        except Exception:
            continue
    return None


def run_pipeline_for_ticker(
    ticker: str,
    aporte: float,
    start: str,
    *,
    end: str | None = None,
    profile: str = "B",  # A ou B
    ref_month: Optional[str] = None,
) -> dict:
    # Config de perfil
    lead = 2
    trend_sma = 100
    dyn_k = 0.8
    exp_thresh = 0.0
    atr_window = 14
    atr_k = 1.5
    risk = 0.005 if profile.upper() == "B" else None

    close = get_prices([ticker], start=start, end=end).sort_index()
    if close.empty:
        raise ValueError("Sem dados para o período selecionado.")
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

    last_close_price = _to_float(close[ticker].iloc[-1]) if ticker in close.columns else None

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
            "data": {"tickers": [ticker], "start": start, "end": end},
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

    trades_columns: list[str] = []
    trades_preview: list[dict[str, object]] = []
    trades_df: pd.DataFrame | None = None
    filtered_trades_df: pd.DataFrame | None = None
    if trades_dst.exists():
        try:
            trades_df = pd.read_csv(trades_dst)
            trades_columns = list(trades_df.columns)
            if ref_month:
                try:
                    dt_ref = pd.to_datetime(ref_month + "-01")
                    next_month = (dt_ref + pd.offsets.MonthBegin(1))
                    mask = pd.to_datetime(trades_df["Exit Timestamp"], errors="coerce")
                    filtered_trades_df = trades_df[(mask >= dt_ref) & (mask < next_month)]
                except Exception:
                    filtered_trades_df = trades_df
            if filtered_trades_df is None:
                filtered_trades_df = trades_df
            # Tornar mais claro: se Status != Closed, esconder Exit Timestamp na prévia
            try:
                _prev = filtered_trades_df.copy()
                if "Status" in _prev.columns and "Exit Timestamp" in _prev.columns:
                    mask_open = _prev["Status"].astype(str).str.lower() != "closed"
                    _prev.loc[mask_open, "Exit Timestamp"] = ""
                trades_preview = _prev.head(10).to_dict(orient="records")
            except Exception:
                trades_preview = filtered_trades_df.head(10).to_dict(orient="records")
        except Exception:
            trades_columns = []
            trades_preview = []

    # Painel de operações
    portfolio_summary = {
        "current_value": None,
        "pnl": None,
        "pnl_pct": None,
        "peak_value": None,
        "min_value": None,
    }
    current_trade = None
    open_summary = {"count": 0, "pnl": 0.0, "return_pct": None}
    closed_summary = {"count": 0, "pnl": 0.0, "return_pct": None}
    last_closed_trade = None
    status = {"code": "waiting", "label": "Aguardando primeiro sinal", "detail": "Nenhuma operação executada dentro do intervalo selecionado."}

    pf_ticker = pf.get(ticker)
    if pf_ticker is not None:
        equity = _series_from_pf(pf_ticker)
        if equity is not None:
            equity = equity.astype(float)
            equity = equity.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if not equity.empty:
                current_value = _to_float(equity.iloc[-1])
                peak_value = _to_float(equity.max())
                min_value = _to_float(equity.min())
                if current_value is not None:
                    pnl = current_value - aporte
                    pnl_pct = (pnl / aporte * 100.0) if aporte else None
                    portfolio_summary.update(
                        {
                            "current_value": current_value,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "peak_value": peak_value,
                            "min_value": min_value,
                        }
                    )

    if portfolio_summary["current_value"] is None and row:
        total_return_pct = _to_float(row.get("Total Return [%]"))
        if total_return_pct is not None:
            current_value = aporte * (1.0 + total_return_pct / 100.0)
            portfolio_summary.update(
                {
                    "current_value": current_value,
                    "pnl": current_value - aporte,
                    "pnl_pct": total_return_pct,
                    "peak_value": None,
                    "min_value": None,
                }
            )

    analytics_df = filtered_trades_df if filtered_trades_df is not None else trades_df

    if analytics_df is not None and not analytics_df.empty:
        open_trades = analytics_df.loc[analytics_df.get("Status").fillna("") != "Closed"].copy()
        closed_trades = analytics_df.loc[analytics_df.get("Status").fillna("") == "Closed"].copy()

        if not open_trades.empty:
            open_summary["count"] = int(len(open_trades))
            open_pnl_val = _to_float(open_trades.get("PnL").sum())
            open_summary["pnl"] = open_pnl_val if open_pnl_val is not None else 0.0
            open_ret = open_trades.get("Return")
            if open_ret is not None and not open_ret.dropna().empty:
                open_summary["return_pct"] = _to_float(open_ret.mean() * 100.0)

            latest_open = open_trades.sort_values("Entry Timestamp").iloc[-1]
            latest_open_ret = _to_float(latest_open.get("Return"))
            current_trade = {
                "entry_ts": latest_open.get("Entry Timestamp"),
                "size": _to_float(latest_open.get("Size")),
                "entry_price": _to_float(latest_open.get("Avg Entry Price")),
                "last_price": _to_float(latest_open.get("Avg Exit Price")) or last_close_price,
                "pnl": _to_float(latest_open.get("PnL")),
                "return_pct": latest_open_ret * 100.0 if latest_open_ret is not None else None,
                "status": latest_open.get("Status", "Open"),
            }
            status = {
                "code": "open",
                "label": "Posição aberta",
                "detail": f"Existem {open_summary['count']} operações em andamento. Última entrada em {current_trade['entry_ts']}.",
            }

        if not closed_trades.empty:
            closed_summary["count"] = int(len(closed_trades))
            closed_pnl_val = _to_float(closed_trades.get("PnL").sum())
            closed_summary["pnl"] = closed_pnl_val if closed_pnl_val is not None else 0.0
            closed_ret = closed_trades.get("Return")
            if closed_ret is not None and not closed_ret.dropna().empty:
                closed_summary["return_pct"] = _to_float(closed_ret.mean() * 100.0)

            last_closed = closed_trades.sort_values("Exit Timestamp").iloc[-1]
            last_closed_ret = _to_float(last_closed.get("Return"))
            last_closed_trade = {
                "entry_ts": last_closed.get("Entry Timestamp"),
                "exit_ts": last_closed.get("Exit Timestamp"),
                "pnl": _to_float(last_closed.get("PnL")),
                "return_pct": last_closed_ret * 100.0 if last_closed_ret is not None else None,
                "direction": last_closed.get("Direction"),
            }
            if status["code"] != "open":
                status = {
                    "code": "flat_after_trades",
                    "label": "Sem posição – trades encerradas",
                    "detail": f"{closed_summary['count']} operações concluídas no período. Aguarde novos sinais.",
                }
    elif row:
        status = {
            "code": "waiting",
            "label": "Aguardando primeiro sinal",
            "detail": "Nenhum trade foi identificado para este ticker dentro do intervalo analisado.",
        }

    return {
        "ticker": ticker,
        "summary_path": str(report_path),
        "trades_path": str(trades_dst) if trades_dst.exists() else None,
        "row": row,
        "ts": ts,
        "start": start,
        "end": end,
        "aporte": aporte,
        "profile": profile,
        "trades_columns": trades_columns,
        "trades_preview": trades_preview,
        "portfolio_summary": portfolio_summary,
        "current_trade": current_trade,
        "open_summary": open_summary,
        "closed_summary": closed_summary,
        "last_closed_trade": last_closed_trade,
        "status": status,
        "ref_month": ref_month,
        "filtered_trades_count": int(len(analytics_df)) if analytics_df is not None else 0,
        "last_bar": str(close.index.max().date()) if not close.empty else None,
    }


@APP.route("/", methods=["GET", "POST"])
def index():
    tickers = _load_universe()
    today = datetime.today().date()
    # Padrão: último dia útil (T-1)
    try:
        default_end_ts = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[0]
        default_end = default_end_ts.strftime("%Y-%m-%d")
    except Exception:
        default_end = today.strftime("%Y-%m-%d")
    default_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    current_month = today.strftime("%Y-%m")

    form_state = {
        "ticker": tickers[0] if tickers else "",
        "aporte": "100000",
        "start": default_start,
        "end": default_end,
        "profile": "B",
        "period": "custom",
        "ref_month": current_month,
    }

    result = None
    if request.method == "POST":
        form_state.update(
            {
                "ticker": request.form.get("ticker") or form_state["ticker"],
                "aporte": request.form.get("aporte") or form_state["aporte"],
                "start": request.form.get("start") or form_state["start"],
                "end": request.form.get("end") or form_state["end"],
                "profile": request.form.get("profile") or form_state["profile"],
                "period": request.form.get("period") or "custom",
                "ref_month": request.form.get("ref_month") or form_state["ref_month"],
            }
        )

        ticker = form_state["ticker"]
        aporte_raw = form_state["aporte"]
        start = form_state["start"]
        end = form_state["end"] or None
        profile = form_state["profile"]
        ref_month = form_state["ref_month"]

        try:
            aporte_val = float(aporte_raw or 100000)
        except ValueError:
            flash("Valor de aporte inválido.", "error")
            aporte_val = None

        if aporte_val is not None:
            if start and end:
                try:
                    start_dt = datetime.fromisoformat(start)
                    end_dt = datetime.fromisoformat(end)
                    if start_dt >= end_dt:
                        flash("A data inicial deve ser anterior à data final.", "error")
                        aporte_val = None
                except ValueError:
                    flash("Datas inválidas.", "error")
                    aporte_val = None

        if aporte_val is not None and ticker:
            try:
                result = run_pipeline_for_ticker(
                    ticker=ticker,
                    aporte=aporte_val,
                    start=start,
                    end=end,
                    profile=profile,
                    ref_month=ref_month,
                )
                # manter cache ordenado por ticker recente
                if ticker in LATEST_RESULTS:
                    del LATEST_RESULTS[ticker]
                LATEST_RESULTS[ticker] = result
                while len(LATEST_RESULTS) > MAX_RESULTS:
                    LATEST_RESULTS.popitem(last=False)
                flash(f"Execução concluída para {ticker}", "success")
            except Exception as exc:
                msg = str(exc)
                if "Time series is too short" in msg:
                    msg = (
                        "Série muito curta para treinar. Amplie as datas (ex.: 3 meses) ou reduza o input_size."
                    )
                flash(f"Falha na execução: {msg}", "error")
    # últimas execuções do registry
    try:
        recent = last_runs("reports/experiments.sqlite", limit=15)
    except Exception:
        recent = pd.DataFrame()
    return render_template(
        "index.html",
        tickers=tickers,
        result=result,
        results=list(reversed(list(LATEST_RESULTS.values()))),
        recent=recent,
        form=form_state,
    )


def create_app() -> Flask:
    return APP


if __name__ == "__main__":
    debug_flag = os.getenv("FLASK_DEBUG", "0") in {"1", "true", "True"}
    port = int(os.getenv("PORT", "5000"))
    APP.run(host="0.0.0.0", port=port, debug=debug_flag)
