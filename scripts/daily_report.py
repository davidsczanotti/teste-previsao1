from __future__ import annotations

"""
scripts/daily_report.py

Gera um relatório diário a partir do registry SQLite (reports/experiments.sqlite),
consolidando runs e métricas do dia em CSVs dentro de reports/daily/YYYY-MM-DD/.

Uso:
    poetry run python -m scripts.daily_report --day 2025-09-25
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera relatório diário do registry")
    p.add_argument("--db", type=str, default="reports/experiments.sqlite")
    p.add_argument("--day", type=str, default=None, help="YYYY-MM-DD em UTC (default: hoje)")
    p.add_argument("--limit", type=int, default=1000, help="Limite de linhas por consulta")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Registry não encontrado em {db_path}")

    day = args.day or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    outdir = Path(f"reports/daily/{day}")
    outdir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        runs_q = (
            """
            SELECT run_id, created_at, name, notes, git_hash, report_path,
                   json_extract(config_json, '$.data.tickers') AS tickers
            FROM runs
            WHERE substr(created_at, 1, 10) = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
        )
        runs_df = pd.read_sql_query(runs_q, conn, params=(day, args.limit))
        runs_df.to_csv(outdir / "last_runs.csv", index=False)

        metrics_q = (
            """
            SELECT r.created_at, r.name, r.notes, m.ticker,
                   m.total_return, m.sharpe, m.win_rate, m.max_drawdown, m.trades,
                   m.run_id
            FROM metrics m
            JOIN runs r ON r.run_id = m.run_id
            WHERE substr(r.created_at, 1, 10) = ?
            ORDER BY r.created_at DESC
            LIMIT ?
            """
        )
        metrics_df = pd.read_sql_query(metrics_q, conn, params=(day, args.limit))
        metrics_df.to_csv(outdir / "metrics_by_ticker.csv", index=False)

    if not metrics_df.empty:
        agg = (
            metrics_df.groupby("ticker", as_index=False)
            .agg(
                avg_return=("total_return", "mean"),
                avg_sharpe=("sharpe", "mean"),
                avg_win_rate=("win_rate", "mean"),
                avg_mdd=("max_drawdown", "mean"),
                total_trades=("trades", "sum"),
                runs=("run_id", "nunique"),
            )
            .sort_values(["avg_sharpe", "avg_return"], ascending=[False, False])
        )
        agg.to_csv(outdir / "top_tickers.csv", index=False)

        # Pequeno sumário textual
        topn = agg.head(10)
        lines = []
        lines.append(f"Dia: {day}")
        lines.append(f"Runs no dia: {len(runs_df)}")
        lines.append(f"Tickers distintos: {agg.shape[0]}")
        lines.append("Top 5 por Sharpe médio:")
        for _, row in topn.head(5).iterrows():
            lines.append(
                f" - {row['ticker']}: Sharpe={row['avg_sharpe']:.2f}, Ret={row['avg_return']:.2f}%, Trades={int(row['total_trades'])}"
            )
        (outdir / "SUMMARY.txt").write_text("\n".join(lines))

    print(f"[daily_report] Relatórios salvos em {outdir}")


if __name__ == "__main__":
    main()

