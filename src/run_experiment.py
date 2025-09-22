# src/run_experiment.py
import pandas as pd
from ingest import get_prices
from features import add_ta
from labeling import triple_barrier_labels
from backtest import run_backtest
from models_ts import build_neuralforecast, nhits_cfg

# 1) Dados
close = get_prices()
feats = add_ta(close)

# 2) Preparar dados para NeuralForecast (formato longo)
# O target (y) será o retorno percentual do dia seguinte.
returns = close.pct_change().shift(-1)
df_long = returns.unstack().reset_index()
df_long.columns = ["unique_id", "ds", "y"]  # Agora 'y' é o retorno
df_long["ds"] = pd.to_datetime(df_long["ds"])
df_long = df_long.dropna()

# 3) Treinar N-HiTS para prever o retorno
#    Vamos usar o preço (y) como target e a biblioteca cuidará da normalização.
#    O modelo preverá o preço 'h' passos à frente.

horizon = 5  # Prever 5 dias à frente

# Configura o modelo N-HiTS
models = [nhits_cfg()]  # Podemos adicionar nbeats_cfg() e patchtst_cfg() aqui

# Constrói e treina o modelo
nf = build_neuralforecast(models=models, freq="B")  # 'B' para business days
nf.fit(df=df_long)

# Gera previsões
predictions = nf.predict()
predictions = predictions.reset_index()

# 4) Gerar sinais (entries/exits) a partir das previsões
#    Estratégia: comprar se o retorno previsto para o próximo dia for positivo.

preds_petr4 = predictions[predictions["unique_id"] == "PETR4.SA"].set_index("ds")

# Sinal de compra: se a previsão de retorno (NHITS) for maior que um limiar (ex: 0).
trade_threshold = 0
entries_partial = preds_petr4["NHITS"] > trade_threshold
# Sinal de saída: quando a condição de entrada não for mais verdadeira.
exits_partial = ~entries_partial

# --- CORREÇÃO: Alinhar os sinais com o histórico de preços completo ---
# Reindexa os sinais para terem o mesmo índice que a série de preços,
# preenchendo os valores ausentes (onde não há previsão) com False.
entries = entries_partial.reindex(close.index, fill_value=False)
exits = exits_partial.reindex(close.index, fill_value=False)

# 5) Backtest
print("--- Executando Backtest para PETR4.SA ---")
pf = run_backtest(close["PETR4.SA"], entries, exits)
print(pf.stats())
