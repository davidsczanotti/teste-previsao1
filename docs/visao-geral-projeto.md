Visão geral do projeto de previsão & backtest (NHITS + vectorbt)

Objetivo
Construir um pipeline simples e reprodutível para:
(1) baixar cotações, (2) preparar séries longas, (3) treinar um modelo de previsão (NHITS),
(4) transformar previsões em sinais, e (5) rodar um backtest limpo, sem look-ahead.

Arquitetura atual (camadas e arquivos)

Ingestão

src/ingest.py: baixa preços (colunas por ticker, índice datetime).
Saída típica: close_wide (DataFrame com tickers nas colunas).

Preparação

src/prep.py: transforma close_wide em long_df com colunas ['unique_id','ds','y'] e adiciona features simples quando necessário.

Modelagem

src/models_ts.py: função train_predict_nhits(...) que:

aceita df_long ou long_df (compatibilidade),

parâmetros como horizon, n_windows, input_size, max_steps, random_seed, start_padding_enabled,

tolera extras via **_ (ex.: verbose, use_gpu) para não quebrar,

executa NeuralForecast.cross_validation, com fallback em séries curtas,

retorna yhat_df no formato ['unique_id','ds','y_hat'].

Sinais

src/signals.py:

alinha previsão y_hat ao índice de preços,

cria condições de entries/exits (ex.: divergência prevista x preço, filtros de tendência, etc.),

não aplica o “executar no próximo candle” aqui (ver decisão abaixo) — isso é feito globalmente no backtest.

Backtest

src/backtest.py:

funções utilitárias para:

garantir Series booleanas alinhadas ao índice,

aplicar shift(1) em entries/exits (execução no próximo candle) — evita look-ahead,

montar Portfolio do vectorbt com accumulate=True,

calcular métricas de forma robusta a diferenças de versão (total return, sharpe, max drawdown em % positivo, win rate, #trades),

salvar trades_{ticker}.csv e summary_baseline.csv.

Interface: run_backtest(close_wide, signals, fees, slippage, init_cash, ...).

Experimento fim-a-fim

src/run_experiment.py: CLI com argumentos (tickers, data inicial, horizon, janelas, input_size, fees, slippage, init_cash, etc.) que:

ingere preços,

prepara long_df,

treina/prediz (models_ts),

gera sinais (signals),

roda backtest (backtest) e salva relatórios.

Teste rápido (anti-regressão)

scripts/test_backtest_smoke.py: valida o shift(1) e métricas básicas. Deve imprimir “OK ✅ …”.

Decisões importantes (e por quê)
1) Look-ahead — executar entradas/saídas no próximo candle

Problema identificado: sinais eram executados no mesmo bar em que eram gerados → performance artificial (muito alta).

Decisão: centralizar o atraso de execução no backtest:

Em run_backtest, aplicamos entries = entries.shift(1) e exits = exits.shift(1).

Motivo: manter isso num único lugar evita duplicidade e erros (se o sinal já vier com shift, dá duplicado).

Alternativa rejeitada: aplicar o shift(1) dentro de signals.py.

Rejeitamos para não espalhar responsabilidade. O backtest é “a verdade de execução”.

2) Alinhamento seguro & índices duplicados

Encontramos ValueError: cannot reindex on an axis with duplicate labels.

Causa: Series com índice duplicado (datas repetidas) sendo reindexadas.

Decisão: sempre sanitizar antes de reindex:
series = series[~series.index.duplicated(keep='last')].reindex(target_index)

Alternativa rejeitada: reindex direto (gera erro em pandas recentes).

3) Reindexamento e tipos

Alguns sinais vinham como object/int.

Decisão: _ensure_bool_series converte para bool e reindex(fill_value=False).

4) Métricas robustas a versões do vectorbt

Versões diferentes exigem freq para Sharpe/Sortino; às vezes mudam nomes de colunas.

Decisão: criar “getters” tolerantes (_safe_total_return, _safe_sharpe, _safe_max_dd, _win_rate_and_trades) e:

retornar Max Drawdown em % positivo (sem sinal invertido),

Win Rate via records_readable (ou records bruto se necessário).

Alternativa rejeitada: depender só de pf.stats(); em algumas versões, colunas/nomenclaturas mudam.

5) NHITS & séries curtas

Em janelas curtas recebemos: “Time series is too short for training”.

Decisão: train_predict_nhits aceita start_padding_enabled=True (e fallback reduzindo input_size).

Alternativa rejeitada: forçar input_size fixo alto (quebra para janelas curtas).

6) Compatibilidade de parâmetros em models_ts

run_experiment passou random_seed, use_gpu, verbose, start_padding_enabled em diferentes momentos.

Decisão: train_predict_nhits(...) dá suporte a esses parâmetros e ignora extras via **_.

7) SMA/Tendência por ticker

Problema: KeyError: 'BOVA11.SA' quando se usava uma única série em lugar de um dict por ticker.

Decisão: onde houver filtros por tendência, usar dict por ticker ou aplicar por coluna no close_wide antes de “desempilhar”.

8) Smoke test

Erro inicial: script esperando coluna Entry Time (dependente de versão).

Decisão: tornar o script tolerante ao nome da coluna (ex.: Entry Time / Entry Open / etc.) e focar no efeito do shift(1) e no resumo.

O que testamos e rejeitamos

Executar sinais no mesmo candle
→ Rejeitado: gera look-ahead e resultados irreais (ex.: “Sharpe” altíssimo e retornos absurdos).

Reindex sem tratar duplicados
→ Rejeitado: pandas moderno levanta erro; precisamos normalizar o índice.

Calcular métricas só por pf.stats()
→ Rejeitado: quebrou a portabilidade entre versões e às vezes retornava NaN para Sharpe (sem freq).

Apertar demais input_size sem padding
→ Rejeitado: falhava em janelas curtas. Mantivemos fallback com start_padding_enabled=True e input_size reduzido quando necessário.

Aplicar shift(1) em signals.py
→ Rejeitado: fácil esquecer e aplicar duas vezes; melhor concentrar isso no backtest.

Resultados típicos (padrões vistos)

Antes do shift(1): retornos e Sharpe inflados (alguns rodavam com +80%/+100% no período).

Após o shift(1): resultados mais realistas. ITUB4.SA frequentemente bem, BOVA11.SA ok, PETR4.SA e VALE3.SA variando com o limiar de sinais.

Ao alinhar sinais e remover duplicidade de índice, os erros de reindex sumiram.

Como rodar (atalho)
# ambiente ativo (poetry shell / venv)
poetry run python -m src.run_experiment \
  --tickers VALE3.SA PETR4.SA BOVA11.SA ITUB4.SA \
  --start 2020-01-01 \
  --horizon 5 \
  --n-windows 8 \
  --input-size 60 \
  --max-steps 300 \
  --fees 0.0005 --slippage 0.0005 --init-cash 100000


Saídas:

reports/summary_baseline.csv (resumo por ticker)

reports/trades_{TICKER}.csv (trades legíveis)

Logs DEBUG {ticker}: entries=... exits=...

Teste rápido do backtest (confirma shift(1)):

poetry run python -m scripts.test_backtest_smoke
# Deve imprimir: "OK ✅ Shift(1) aplicado corretamente..."

Pontos de atenção

Sharpe/Sortino: se precisar sempre populados, podemos setar freq='B' explicitamente onde a versão exigir (hoje já tratamos robustamente; quando a versão não provê, deixamos NaN).

NIXTLA_ID_AS_COL (FutureWarning): dá para silenciar ajustando env var NIXTLA_ID_AS_COL=1 ou deixando como está (não afeta backtest).

Séries curtas: use --input-size menor ou --start mais antigo; nosso models_ts já tenta um fallback automático.

Próximos passos (roadmap)

Calibração de sinais (curva ROC de limiares)

Ex.: entrar quando y_hat[t+h] / close[t] - 1 > τ e sair no oposto; varrer τ (0.1%…1.5%).

Guardar tabela de desempenho por τ e por ticker.

Regras de posição e risco

Tamanho fixo vs. vol targeting; stop-loss/time-stop; travas para evitar “overtrade”.

Features exógenas (X)

Volatilidade implícita (se disponível), commodities (ex.: minério para VALE), CDI/câmbio como contexto, lags.

Ensembles e modelos alternativos

Comparar NHITS vs. NBEATS/MLP/LightGBM de features manuais; média/mediana de previsões; model confidence como filtro.

Validação walk-forward com “refit”

Re-treinar periodicamente (mensal/quinzenal), guardando um roll de hiperparâmetros vencedores.

Reprodutibilidade & tracking

Salvar args.json, metrics.json, seed, commit git, e logs por run (ex.: runs/YYYYMMDD_HHMM).

Relatório HTML

Um notebook/HTML com gráficos de curva de capital, distribuição de retornos, heatmaps de τ, etc.

“Por que estamos no caminho certo?”

Saneamento dos fundamentos concluído:

Sem look-ahead (execução no próximo candle no backtest),

Alinhamento/rindex seguro,

Métricas consistentes em várias versões do vectorbt,

Função de modelo robusta a séries curtas e a parâmetros extras.

Com a base correta, agora faz sentido otimizar/regulamentar estratégia (limiares, risco, custos) sem distorções.

Anexos (referência rápida)
Formato de long_df
unique_id, ds, y
VALE3.SA, 2024-01-02, 56.12
VALE3.SA, 2024-01-03, 56.80
...

Saída de previsões (yhat_df)
unique_id, ds, y_hat
VALE3.SA, 2024-02-01, 57.45
VALE3.SA, 2024-02-02, 57.51
...

Resumo de backtest (exemplo)
           Total Return [%]  Sharpe Ratio  Win Rate [%]  Max Drawdown [%]  Trades
ITUB4.SA            22.33         0.64         50.00             8.19        40
...