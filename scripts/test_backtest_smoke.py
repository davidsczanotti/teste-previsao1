# scripts/test_backtest_smoke.py
import pandas as pd
import numpy as np

from src.backtest import run_backtest

# ----- Dados de teste (10 dias úteis) -----
dates = pd.date_range("2024-01-01", periods=10, freq="B")

# Preços "subindo" só para facilitar a leitura
close = pd.Series(np.linspace(100, 110, len(dates)), index=dates, name="TESTE.SA")
close_wide = close.to_frame()

# Sinais: marcar entrada no D2 e saída no D4
# Como run_backtest aplica shift(1), a execução ocorrerá em D3 e D5
entries = pd.Series(False, index=dates)
exits = pd.Series(False, index=dates)
D2, D4 = dates[2], dates[4]
entries.loc[D2] = True
exits.loc[D4] = True

signals = {"TESTE.SA": {"entries": entries, "exits": exits}}

# ----- Rodar backtest -----
summary, portfolios = run_backtest(
    close_wide=close_wide,
    signals=signals,
    init_cash=10_000.0,
    fees=0.0,
    slippage=0.0,
    direction="longonly",
    save_trades=False,
    report_path="reports/summary_smoke.csv",
)

print("\n=== SMOKE SUMMARY ===")
print(summary)

# ----- Verificações -----
pf = portfolios["TESTE.SA"]

# Deve haver 1 trade
n_trades = len(pf.trades.records)
assert n_trades == 1, f"Esperava 1 trade, veio {n_trades}"

# Buscar entrada/saída de forma robusta, cobrindo diferentes versões do vectorbt
entry_dt = None
exit_dt = None

# 1) Tentar 'records_readable' com diferentes nomes de colunas
try:
    rec = pf.trades.records_readable  # DataFrame
    # Candidatos por tempo
    time_pairs = [
        ("Entry Time", "Exit Time"),
        ("Entry Timestamp", "Exit Timestamp"),
        ("Entry Date", "Exit Date"),
    ]
    for ec, xc in time_pairs:
        if ec in rec.columns and xc in rec.columns:
            entry_dt = pd.to_datetime(rec[ec].iloc[0])
            exit_dt = pd.to_datetime(rec[xc].iloc[0])
            break
    # Se não achou tempo, tentar pares por índice
    if entry_dt is None:
        index_pairs = [
            ("Entry Index", "Exit Index"),
            ("Entry Idx", "Exit Idx"),
            ("Entry Row", "Exit Row"),
        ]
        for ec, xc in index_pairs:
            if ec in rec.columns and xc in rec.columns:
                entry_dt = close.index[int(rec[ec].iloc[0])]
                exit_dt = close.index[int(rec[xc].iloc[0])]
                break
except Exception:
    pass

# 2) Fallback: usar 'records' (array estruturado) e mapear índices -> datas
if entry_dt is None:
    raw = pf.trades.records
    names = raw.dtype.names

    def pick(*cands):
        for c in cands:
            if c in names:
                return c
        return None

    efield = pick("entry_idx", "EntryIdx", "entry_index")
    xfield = pick("exit_idx", "ExitIdx", "exit_index")
    assert efield and xfield, f"Não achei campos de índice nas trades: {names}"

    entry_dt = close.index[int(raw[0][efield])]
    exit_dt = close.index[int(raw[0][xfield])]

# Esperados (por causa do shift(1) aplicado no run_backtest)
expected_entry = dates[3]
expected_exit = dates[5]

assert entry_dt == expected_entry, f"Entrada esperada {expected_entry.date()} vs {entry_dt}"
assert exit_dt == expected_exit, f"Saída esperada {expected_exit.date()} vs {exit_dt}"

# Max Drawdown pode sair NaN nessa série curtinha; quando existir, deve ser >= 0
mdd = summary.loc["TESTE.SA", "Max Drawdown [%]"]
assert pd.isna(mdd) or mdd >= 0, f"MDD deveria ser >=0 ou NaN. Veio {mdd}"

print("\nOK ✅  Shift(1) aplicado corretamente e métricas básicas OK.")
