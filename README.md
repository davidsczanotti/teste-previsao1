# teste-previsao1 — Previsão (NHITS) + Backtest (vectorbt) + Web

Pipeline completo para baixar preços, treinar NHITS (NeuralForecast), gerar sinais e rodar backtest sem look‑ahead. Inclui frontend Flask com modo Paper‑Live, agendador e página de configuração.

---

## Sumário rápido
- Ingestão: `src/ingest.py` → `close_wide` (tickers em colunas).
- Preparação: `src/prep.py` → `long_df` (`unique_id`, `ds`, `y`).
- Modelo: `src/models_ts.py` → `train_predict_nhits(...)` (retorna `yhat_df`).
- Sinais: `src/signals.py` → `entries/exits` por ticker (sem shift aqui).
- Backtest: `src/backtest.py` → aplica `shift(1)` (T+1), cria portfólios e métricas tolerantes a versão.
- Web: `scripts/web_app.py` (Paper‑Live, pré‑cheque, config, agendador, reset de Go‑Live).
- Jobs: `scripts/daily_jobs.py` (manhã/EOD) + `scripts/run_universe_scan.py` (ranking).

Saídas web: `reports/web_summary_{TICKER}_{TS}.csv`, `reports/web_trades_{TICKER}_{TS}.csv`.

---

## Docker (servidor)

```bash
docker build -t teste-previsao1:latest .
docker run -d --name previsao-web -p 5000:5000 -v $(pwd)/reports:/app/reports teste-previsao1:latest
```

Acesse `http://SEU_SERVIDOR:5000`. Página de configurações: `/config`.

Agendador e status (polling) também em `/config`.

---

## Guia rápido (check‑list)

1) Suba o container e acesse `/config`.
   - Habilite o agendador e defina `am_time` (ex.: 08:00) e `eod_time` (ex.: 20:05).
   - Defina o universo (arquivo) e os defaults: `NHITS – modo = Completo`, `Contexto = Automático`, `Ajuste automático = ON` e `min_input_size ≥ 30`.
   - Salve e, se quiser, clique “Executar agora (EOD)” para testar. Acompanhe o status em tempo real.

2) Na página inicial:
   - Selecione o ticker, mantenha “Contexto de treino = Automático” e “NHITS – modo = Completo”.
   - Deixe “Paper‑Live” ligado. Defina o aporte e o mês de referência.
   - Clique “Executar simulação”.

3) Validação rápida no card de resultado:
   - Verifique o “Pré‑cheque”: `Barras X/Y (OK)` e “Paper‑Live (ON)”.
   - Confira “Dados até: AAAA‑MM‑DD (último pregão)” e “Go‑Live”.
   - Se houver “Compra agendada…”, significa que havia sinal em T−1; a execução aparece no pregão seguinte.

4) Auditoria e histórico:
   - Resumo e trades do Web: `reports/web_summary_{TICKER}_{TS}.csv` e `reports/web_trades_{TICKER}_{TS}.csv`.
   - Ranking do universo: `reports/universe_rankings.csv` (e cópias diárias em `reports/daily/YYYY‑MM‑DD/`).
   - Registry local: `reports/experiments.sqlite`.

5) Reexecuções diárias:
   - O agendador roda os jobs nos horários configurados; os campos de status em `/config` atualizam por polling.
   - Você pode resetar o Go‑Live por ticker (botão no card) ou “Zerar todos” em `/config` para recomeçar os testes.

Boas práticas:
- Prefira 12–24 meses de contexto (Automático). Use “Rápido” apenas para diagnóstico.
- Mantenha “Paper‑Live” ON para não herdar trades históricos.
- Ajuste o risco (perfil B) se o aporte for baixo e o tamanho de posição estiver zerando.

---

## Web App (Paper‑Live)
- Treina com histórico, mas opera só a partir do Go‑Live (último pregão ao clicar). Sem herdar trades passados.
- “Compra agendada” quando há `entry` em T−1 (execução em T+1 garantida pelo backtest).
- Contexto de treino seguro: Automático (12–24 meses) com pré‑cheque de barras mínimas. Bloqueio de execuções fracas no modo básico.
- NHITS – modos: Completo (n_windows=24) e Rápido (n_windows=12).
- Ajustes automáticos (opcionais): quando faltar histórico, o app tenta expandir período e reduzir `input_size` (limite `min_input_size`) e `n_windows`.
- Relatórios do usuário: `reports/web_*`; Go‑Live por ticker em `reports/user_state.sqlite` (reset por ticker ou “zerar todos” em `/config`).

Preset recomendado: Contexto = Automático; NHITS = Completo; Paper‑Live ON; Perfil B (risco 0,5%); Ref. mensal = mês corrente.

---

## Capturas de tela

As imagens a seguir ilustram a UI atual (caso deseje anexar seus próprios screenshots, salve‑os em `docs/img/` e atualize os caminhos):

- Página inicial (execução por ticker): `docs/img/web-index.png`
- Configurações e agendador (status em tempo real): `docs/img/web-config.png`

---

## Instalação local (opcional)
```bash
git clone <SEU_FORK_OU_REPO>
cd teste-previsao1
poetry install
```

### CLI (experimentos)
```bash
poetry run python -m src.run_experiment \
  --tickers VALE3.SA PETR4.SA BOVA11.SA ITUB4.SA \
  --start 2020-01-01 \
  --horizon 5 --n-windows 8 --input-size 60 --max-steps 300 \
  --fees 0.0005 --slippage 0.0005 --init-cash 100000
```

---

## Garantias anti look‑ahead
- O atraso de execução é aplicado no backtest:
  ```python
  entries = entries.shift(1, fill_value=False)
  exits   = exits.shift(1, fill_value=False)
  ```
- Não aplicamos `shift(1)` em `signals.py`.

---

## Agendador & Config (/config)
- Persistência em `reports/app_config.sqlite` (sem .env):
  - Horários `am_time` (manhã) / `eod_time` (pós‑fechamento)
  - Universo (arquivo), modo NHITS padrão (fast/full)
  - Contexto (auto/6/12/24), auto‑ajuste e `min_input_size`
- Ações: “Executar agora (manhã/EOD)”, “Executar scan”, “Zerar todos os Go‑Live”.
- Status por polling: fila, executando, último job e links para `reports/universe_rankings.csv` (e cópia diária).

---

## Registro de experimentos (SQLite)
Habilite no YAML ou via CLI. A UI também registra runs no mesmo SQLite (`reports/experiments.sqlite`).

---

## Sinais e filtros (opcionais)
- Trend/SMA, limiar dinâmico por volatilidade, Bandas de Bollinger, stop por ATR, cooldown e máximo de barras em posição.

---

## Smoke test (anti‑regressão)
```bash
poetry run python -m scripts.test_backtest_smoke
```

---

## Dores e soluções incorporadas
- Índice duplicado → saneamento antes de `reindex`.
- Sinais com dtype estranho → `_ensure_bool_series` força `bool`.
- Séries curtas no NHITS → padding + redução controlada de `input_size`.
- Variação de métricas entre versões do vectorbt → getters tolerantes.
- Operar só após o depósito → Paper‑Live (corte no Go‑Live) + relatórios do usuário separados.

---

## Scripts úteis
- `scripts/web_app.py` – frontend Flask.
- `scripts/daily_jobs.py` – jobs manhã/EOD controlados pela config.
- `scripts/run_universe_scan.py` – scan do universo + ranking.
- `scripts/test_backtest_smoke.py` – sanity do backtest.

---

## FAQ
1) “TypeError … unexpected keyword argument”  
→ Use as assinaturas atuais de `train_predict_nhits` (aceita `**kwargs`).

2) “Sharpe = NaN”  
→ Dependendo da versão do vectorbt, `freq` é necessária. Os getters tolerantes já tratam; pode continuar NaN sem afetar outras métricas.

3) FutureWarning (NIXTLA_ID_AS_COL)  
→ Mensagem da NeuralForecast. Pode ser ignorada; não afeta o pipeline.
