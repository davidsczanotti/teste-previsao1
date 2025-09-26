from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from src.app_config import get_config, set_config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Daily jobs wrapper")
    p.add_argument("--job", choices=["eod", "morning"], required=True)
    args = p.parse_args(argv)

    # EOD: exemplo chama o scan do universo para registrar métricas diárias
    cfg = get_config()
    universe = cfg.get("universe_path", "configs/universe_b3.txt")
    mode = cfg.get("default_mode", "full")
    months_raw = cfg.get("context_months", "auto")
    try:
        months = 12 if months_raw == "auto" else int(months_raw)
    except Exception:
        months = 12

    from datetime import date
    from pandas import DateOffset, Timestamp
    end_dt = date.today()
    start_dt = (Timestamp(end_dt) - DateOffset(months=months)).date()

    if mode == "fast":
        n_windows, step_size, max_steps = 12, 10, 30
    else:
        n_windows, step_size, max_steps = 24, 5, 50
    if args.job == "eod":
        cmd = [
            sys.executable, "-m", "scripts.run_universe_scan",
            "--universe", universe, "--batch-size", "20",
            "--start", str(start_dt),
            "--n-windows", str(n_windows), "--step-size", str(step_size), "--max-steps", str(max_steps),
        ]
    else:
        # morning: reexecuta o scan leve para atualizar ranking / aquecimento de modelos
        cmd = [
            sys.executable, "-m", "scripts.run_universe_scan",
            "--universe", universe, "--batch-size", "20",
            "--start", str(start_dt),
            "--n-windows", str(n_windows), "--step-size", str(step_size), "--max-steps", str(max_steps),
        ]

    print(f"[daily_jobs] running: {' '.join(cmd)}")
    Path("reports").mkdir(parents=True, exist_ok=True)
    # marcar como em execução
    ts_start = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    set_config({"job_running": args.job, "job_running_since": ts_start})
    ret = subprocess.run(cmd, check=False)
    # log in config for UI
    ts = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    if args.job == "eod":
        set_config({"last_eod": ts, "last_eod_rc": str(ret.returncode)})
    else:
        set_config({"last_morning": ts, "last_morning_rc": str(ret.returncode)})

    # caminho do ranking, se existir
    out = Path("reports/universe_rankings.csv")
    kv = {"job_running": "", "job_running_since": "", "last_job_msg": f"{args.job} rc={ret.returncode}"}
    if out.exists():
        kv["last_scan_csv"] = str(out)
    # cópia diária
    try:
        from datetime import datetime, timezone
        day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        dcopy = Path(f"reports/daily/{day}/universe_rankings.csv")
        if dcopy.exists():
            kv["last_scan_csv_daily"] = str(dcopy)
    except Exception:
        pass
    set_config(kv)


if __name__ == "__main__":
    main()
