# Visão geral do projeto (NHITS + vectorbt + Web)

## Objetivo
Pipeline reprodutível para: (1) baixar cotações, (2) preparar séries, (3) treinar NHITS, (4) converter previsões em sinais e (5) backtest sem look‑ahead. A camada Web integra tudo com modo Paper‑Live, agendador e configuração persistente.

## Arquitetura (módulos principais)
- Ingestão: `src/ingest.py` → `close_wide` (tickers em colunas; índice datetime único, ordenado).
- Preparação: `src/prep.py` → `long_df` (`unique_id`, `ds`, `y`) + features leves.
- Modelo: `src/models_ts.py` → `train_predict_nhits(...)` (rolling CV, retorna `yhat_df` padronizado).
- Sinais: `src/signals.py` → regras puras para `entries/exits` (não aplica `shift(1)` aqui).
- Backtest: `src/backtest.py` → `shift(1)` no T+1, `vectorbt.Portfolio.from_signals(accumulate=True)` e métricas tolerantes a versão.
- Web: `scripts/web_app.py` (UI, Paper‑Live, validações, ajustes automáticos, reset de Go‑Live, status em tempo real) e `templates/*.html`.
- Config e estado: `src/app_config.py` (config global em SQLite) e `src/user_state.py` (Go‑Live por ticker, SQLite).
- Jobs: `scripts/daily_jobs.py` (manhã/EOD) e `scripts/run_universe_scan.py` (ranking/scan em batches).

## Fluxo Web (Paper‑Live)
1. O usuário escolhe ticker e aporta; Paper‑Live ON.
2. O app treina NHITS no histórico (Contexto de treino Automático 12–24m ou fixos 6/12/24m) e executa o backtest somente a partir do Go‑Live (T−1) → sem herdar trades antigos.
3. Se havia `entry` em T−1, exibe “Compra agendada”; a execução ocorre em T (T+1 garantido pelo backtest).
4. Relatórios são salvos como `reports/web_summary_{TICKER}_{TS}.csv` e `reports/web_trades_{TICKER}_{TS}.csv`.
5. O Go‑Live fica em `reports/user_state.sqlite` (botões para reset individual e global).

## Guardrails e validações
- Pré‑cheque de barras: calcula `required_bars = max(180, input_size*4, n_windows*step_size + input_size + horizon + 60)` e bloqueia execução fraca no modo básico.
- Ajuste automático (opcional): expande o período, reduz `input_size` até um mínimo e, se necessário, reduz `n_windows`.
- Paper‑Live: força operação após o depósito, relatórios separados e sem contaminação do passado.

## Agendador e Config
- Página `/config` com persistência em `reports/app_config.sqlite` (sem .env):
  - Horários AM/EOD, universo, modo NHITS padrão (fast/full), contexto (auto/6/12/24), auto‑ajuste, `min_input_size`.
  - “Executar agora” (manhã/EOD), “Executar scan do universo”, “Zerar Go‑Live”.
  - Status em tempo real: fila, executando, último job, links para `reports/universe_rankings.csv` e cópia diária.
- O agendador interno dispara `scripts/daily_jobs.py`, que por sua vez chama `scripts/run_universe_scan.py` com parâmetros alinhados à config.

## Contratos de dados (resumo)
- `get_prices(tickers, start, end) -> close_wide` (index único/ordenado, colunas=tickers).
- `prepare_long_and_features(close_wide) -> long_df` (`unique_id`, `ds`, `y`).
- `train_predict_nhits(long_df, ...) -> yhat_df` (`unique_id`, `ds`, `y_hat`).
- `build_signals_from_forecast(yhat_df, close_wide, ...) -> {ticker: {entries, exits}}` (Series booleanas alinhadas a `close_wide.index`).
- `run_backtest(close_wide, signals, ...) -> (summary_df, portfolios)`; aplica `shift(1)` e calcula métricas tolerantes (Sharpe, WinRate, MaxDD positivo, Trades, Retorno%).

## Boas práticas e defaults confiáveis
- Contexto de treino: Automático (ou 12–24 meses); evite janelas curtas.
- NHITS: “Completo” para robustez; “Rápido” para diagnóstico.
- Paper‑Live: ON (operação apenas após o depósito/Go‑Live).
- Perfil B (risco 0,5%): sizing por volatilidade; ajuste se o aporte for baixo e o tamanho zerar.

## Saídas e auditoria
- CSVs “web” por execução (`web_summary_*.csv`, `web_trades_*.csv`) e ranking do universo em `reports/universe_rankings.csv`.
- Histórico de execuções em `reports/experiments.sqlite` (registry local).

## Smoke test
`poetry run python -m scripts.test_backtest_smoke` — valida `shift(1)` e o pipeline mínimo.

