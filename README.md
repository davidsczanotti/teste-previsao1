# teste-previsao1 — Pipeline de previsão (NHITS) + Backtest (vectorbt)

> Pipeline completo para baixar preços, treinar NHITS (NeuralForecast), gerar sinais e rodar backtest **sem look‑ahead** com vectorbt.

---

## Sumário rápido
- **Ingestão:** `src/ingest.py` → DataFrame (`close_wide`) com tickers nas colunas.
- **Preparação:** `src/prep.py` → `long_df` (`unique_id`, `ds`, `y`) e features básicas.
- **Modelo:** `src/models_ts.py` → `train_predict_nhits(...)` treina e produz `yhat_df` (`unique_id`, `ds`, `y_hat`) com *fallback* para séries curtas.
- **Sinais:** `src/signals.py` → transforma `yhat_df` em `entries/exits` por ticker (sem shift).
- **Backtest:** `src/backtest.py` → aplica **shift(1)** (executa no próximo candle), cria portfólios e métricas robustas.
- **Experimento:** `src/run_experiment.py` → orquestra tudo (CLI).
- **Smoke test:** `scripts/test_backtest_smoke.py` → garante o `shift(1)` e métricas básicas.

Saídas principais:
- `reports/summary_baseline.csv` (resumo por ticker)
- `reports/trades_{TICKER}.csv` (lista de trades legível)

---

## Instalação

```bash
# 1) clonar e entrar
git clone <SEU_FORK_OU_REPO>
cd teste-previsao1

# 2) instalar dependências
poetry install  # ou pip install -r requirements.txt (se existir)
```

> **Python**: projeto tem sido usado com Python 3.10.  
> **vectorbt**: versões diferentes podem mudar nomes de colunas/estatísticas; o backtest foi escrito para ser robusto a isso.

---

## Como rodar um experimento

Exemplo (mesmos parâmetros que usamos nos testes recentes):

```bash
poetry run python -m src.run_experiment   --tickers VALE3.SA PETR4.SA BOVA11.SA ITUB4.SA   --start 2020-01-01   --horizon 5   --n-windows 8   --input-size 60   --max-steps 300   --fees 0.0005 --slippage 0.0005 --init-cash 100000
```

Dica: rode primeiro com menos `n-windows` e `max-steps` para ver o *pipeline* funcionando, depois aumente.

---

## Garantias anti look-ahead

- O **atraso de execução** está concentrado no **backtest**:
  - Em `run_backtest(...)` aplicamos:
    ```python
    entries = entries.shift(1, fill_value=False)
    exits   = exits.shift(1, fill_value=False)
    ```
  - Isso garante que o sinal gerado em *t* só é executado em *t+1*.
- Não aplicamos `shift(1)` em `signals.py` para evitar duplicidade.

---

## Interpretação das métricas

- `Total Return [%]`: retorno total do portfólio (robusto a versões).
- `Sharpe Ratio`: calculado de forma tolerante; se a versão do vectorbt exigir `freq`, pode sair `NaN`.
- `Max Drawdown [%]`: **positivo** por convenção (ex.: 12.3 = -12.3%).
- `Win Rate [%]` e `Trades`: a partir de `trades.records_readable` (ou `records` como *fallback*).

Os arquivos `reports/trades_{TICKER}.csv` ajudam a auditar entradas/saídas.

---

## Notebook de relatório rápido

Depois de rodar um experimento, abra e execute:
- `notebooks/quick_report.ipynb`

Ele lê `reports/summary_baseline.csv`, mostra a tabela e plota um gráfico simples
de `Total Return [%]` por ticker.

> **Obs.**: usamos `matplotlib` puro (sem seaborn, sem estilos / cores específicas).

---

## Smoke test (anti-regressão do backtest)

```bash
poetry run python -m scripts.test_backtest_smoke
```

Saída esperada (exemplo):
```
DEBUG TESTE.SA: entries=1 exits=1

=== SMOKE SUMMARY ===
          Total Return [%]  Sharpe Ratio  Win Rate [%]  Max Drawdown [%]  Trades
ticker
TESTE.SA          2.150538      9.062123         100.0               NaN       1

OK ✅  Shift(1) aplicado corretamente e métricas básicas OK.
```

---

## Dores e soluções incorporadas

- **Índice duplicado na reindexação** → normalizamos antes de `reindex`.
- **Sinais com dtype estranho** → `_ensure_bool_series` força `bool`.
- **Séries curtas no NHITS** → `start_padding_enabled=True` + *fallback* de `input_size`.
- **Parâmetros “a mais” no modelo** → `train_predict_nhits` aceita `**_` e ignora os não usados.
- **Métricas variando conforme versão** → *getters* tolerantes.

---

## Roadmap

1. **Varredura de limiar (τ)** para transformar `y_hat` em sinal (grid de thresholds).
2. **Regras de risco**: tamanho de posição, time-stop, stop-loss.
3. **Walk-forward com refit** periódico.
4. **Features exógenas** (câmbio, commodities, CDI).
5. **Relatório HTML completo** com curva de capital, distribuição de retornos e heatmaps de performance por parâmetro.

---

## Perguntas frequentes

**1) “Levei `TypeError: train_predict_nhits() got an unexpected keyword argument …`”**  
→ Atualize `src/models_ts.py` para a versão que aceita `**_` no final da assinatura da função.
Parâmetros aceitos: `df_long`, `horizon`, `n_windows`, `input_size`, `max_steps`,
`random_seed`, `start_padding_enabled`, `use_gpu=False` (opcional).

**2) “Sharpe deu `NaN`”**  
→ Em algumas versões do vectorbt, precisa de `freq`. Nosso *getter* tenta extrair de várias fontes;
se ainda assim vier `NaN`, é comportamento esperado (não afeta outras métricas).

**3) “FutureWarning do NIXTLA_ID_AS_COL”**  
→ Pode ignorar, ou exportar: `export NIXTLA_ID_AS_COL=1`.

---

## Estrutura (esperada)

```
.
├── src/
│   ├── ingest.py
│   ├── prep.py
│   ├── models_ts.py
│   ├── signals.py
│   ├── backtest.py
│   └── run_experiment.py
├── scripts/
│   └── test_backtest_smoke.py
├── reports/
│   ├── summary_baseline.csv
│   └── trades_*.csv
└── notebooks/
    └── quick_report.ipynb
```

---

**Qualquer divergência entre seu repositório e este README**: use este documento como
ilha de referência *ideal* e ajuste os arquivos — ou me diga que eu te mando os arquivos completos atualizados.
