from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from src.app_config import get_config


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
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
