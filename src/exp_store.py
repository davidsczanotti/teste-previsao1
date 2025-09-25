"""Local experiment registry using SQLite.

This module persists each backtest run, including the configuration payload and
per-ticker metrics, so we can inspect past experiments without relying on
external services. It is intended to be a lightweight companion to MLflow.

All timestamps are stored in UTC. The schema keeps runs and metrics in separate
 tables:

- runs(run_id TEXT PRIMARY KEY, created_at TEXT ISO timestamp, name TEXT,
  notes TEXT, git_hash TEXT, config_json TEXT, report_path TEXT)
- metrics(run_id TEXT, ticker TEXT, total_return REAL, sharpe REAL, win_rate REAL,
  max_drawdown REAL, trades INTEGER)
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config import Cfg

_DEFAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT,
    notes TEXT,
    git_hash TEXT,
    config_json TEXT NOT NULL,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    total_return REAL,
    sharpe REAL,
    win_rate REAL,
    max_drawdown REAL,
    trades INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def _ensure_parent(path: Path) -> None:
    """Create parent directories for the SQLite file if missing."""
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def create_schema(db_path: str | Path) -> None:
    """Create SQLite database and schema if it does not exist."""
    path = Path(db_path)
    _ensure_parent(path)
    with sqlite3.connect(path) as conn:
        conn.executescript(_DEFAULT_SCHEMA)


def _ensure_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist in the summary DataFrame."""
    expected = {
        "Total Return [%]",
        "Sharpe Ratio",
        "Win Rate [%]",
        "Max Drawdown [%]",
        "Trades",
    }
    missing = expected - set(summary_df.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {missing}")
    return summary_df


def log_run(
    db_path: str | Path,
    cfg: Cfg,
    summary_df: pd.DataFrame,
    report_path: str | Path,
    git_hash: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Persist a single experiment run and associated metrics.

    Args:
        db_path: Location of the SQLite database.
        cfg: Parsed configuration object used in the run.
        summary_df: DataFrame returned by ``run_backtest``.
        report_path: Path to the summary CSV saved on disk.
        git_hash: Optional git commit hash.
        run_id: Custom run identifier; if not provided a timestamp-based id
            is generated.

    Returns:
        The identifier stored in the ``runs`` table.
    """
    path = Path(db_path)
    _ensure_parent(path)
    create_schema(path)

    summary_df = _ensure_columns(summary_df)

    run_id = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    created_at = datetime.now(tz=timezone.utc).isoformat()
    try:
        config_json = cfg.model_dump_json()
    except AttributeError:
        # Compat with pydantic v1
        config_json = cfg.json(indent=None, sort_keys=True)

    metrics_records: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[int]]] = []
    for ticker, row in summary_df.iterrows():
        metrics_records.append(
            (
                run_id,
                str(ticker),
                float(row["Total Return [%]"]) if pd.notna(row["Total Return [%]"]) else None,
                float(row["Sharpe Ratio"]) if pd.notna(row["Sharpe Ratio"]) else None,
                float(row["Win Rate [%]"]) if pd.notna(row["Win Rate [%]"]) else None,
                float(row["Max Drawdown [%]"]) if pd.notna(row["Max Drawdown [%]"]) else None,
                int(row["Trades"]) if pd.notna(row["Trades"]) else None,
            )
        )

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, created_at, name, notes, git_hash, config_json, report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                cfg.experiment.name,
                cfg.experiment.notes,
                git_hash,
                config_json,
                str(report_path),
            ),
        )
        if metrics_records:
            conn.executemany(
                """
                INSERT INTO metrics (
                    run_id, ticker, total_return, sharpe, win_rate, max_drawdown, trades
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                metrics_records,
            )

    return run_id


def last_runs(db_path: str | Path, limit: int = 50) -> pd.DataFrame:
    """Return the most recent runs joined with aggregated metrics."""
    query = """
    SELECT
        r.run_id,
        r.created_at,
        r.name,
        r.notes,
        r.git_hash,
        r.report_path,
        json_extract(r.config_json, '$.data.tickers') AS tickers,
        COUNT(m.ticker) AS tickers_logged,
        AVG(m.total_return) AS avg_total_return,
        AVG(m.sharpe) AS avg_sharpe,
        AVG(m.win_rate) AS avg_win_rate,
        AVG(m.max_drawdown) AS avg_max_drawdown,
        SUM(m.trades) AS total_trades
    FROM runs r
    LEFT JOIN metrics m ON r.run_id = m.run_id
    ORDER BY r.created_at DESC
    LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(limit,))
    return df
