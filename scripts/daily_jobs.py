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
    if args.job == "eod":
        cmd = [sys.executable, "-m", "scripts.run_universe_scan", "--universe", universe, "--batch-size", "20", "--start", "2018-01-01"]
    else:
        # morning: reexecuta o scan leve para atualizar ranking / aquecimento de modelos
        cmd = [sys.executable, "-m", "scripts.run_universe_scan", "--universe", universe, "--batch-size", "20", "--start", "2018-01-01"]

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
