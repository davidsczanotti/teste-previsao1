"""Utility to inspect the local experiment registry (SQLite)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.exp_store import last_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the local experiment registry")
    parser.add_argument("--db", type=str, default="reports/experiments.sqlite", help="Caminho do banco SQLite")
    parser.add_argument("--limit", type=int, default=50, help="Quantidade de runs recentes")
    parser.add_argument("--output", type=str, default=None, help="Se definido, exporta CSV consolidado")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = last_runs(args.db, limit=args.limit)
    if df.empty:
        print("[registry] Nenhum experimento encontrado.")
        return

    pd.options.display.width = 0
    print("[registry] Últimos runs (média por ticker):")
    print(df)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[registry] Exportado para {output_path}")


if __name__ == "__main__":
    main()
