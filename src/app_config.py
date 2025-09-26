from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

DB_PATH = Path("reports/app_config.sqlite")

DEFAULTS: Dict[str, str] = {
    "scheduler_enabled": "1",
    "eod_time": "20:05",
    "am_time": "08:00",
    "market_open": "10:00",
    "market_close": "18:00",
    "universe_path": "configs/universe_b3.txt",
}


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def get_config() -> Dict[str, str]:
    conn = _conn()
    cur = conn.execute("SELECT key, value FROM app_config")
    rows = dict(cur.fetchall())
    conn.close()
    out = DEFAULTS.copy()
    out.update(rows)
    return out


def set_config(kv: Dict[str, str]) -> None:
    if not kv:
        return
    conn = _conn()
    for k, v in kv.items():
        conn.execute(
            "INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)", (str(k), str(v))
        )
    conn.commit()
    conn.close()

