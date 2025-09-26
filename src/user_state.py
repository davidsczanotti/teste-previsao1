from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


DB_PATH = Path("reports/user_state.sqlite")


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_state (
            ticker TEXT PRIMARY KEY,
            go_live DATE NOT NULL,
            profile TEXT,
            aporte REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    return conn


def get_go_live_date(ticker: str) -> Optional[str]:
    conn = _ensure_db()
    cur = conn.execute("SELECT go_live FROM user_state WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_go_live_date(ticker: str, go_live: str, profile: Optional[str] = None, aporte: Optional[float] = None) -> None:
    conn = _ensure_db()
    conn.execute(
        "INSERT OR REPLACE INTO user_state (ticker, go_live, profile, aporte) VALUES (?, ?, COALESCE(?, profile), COALESCE(?, aporte))",
        (ticker, go_live, profile, aporte),
    )
    conn.commit()
    conn.close()


def reset_go_live(ticker: str) -> None:
    conn = _ensure_db()
    conn.execute("DELETE FROM user_state WHERE ticker = ?", (ticker,))
    conn.commit()
    conn.close()
