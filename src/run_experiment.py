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
# O target será o retorno futuro de 1 dia para simplificar.
# Para prever retornos de 5 dias, o target seria `close.pct_change(5).shift(-5)`
df_long = close.unstack().reset_index()
df_long.columns = ["unique_id", "ds", "y"]
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
#    Estratégia simples: comprar se o preço previsto para D+1 for maior que o de D+0

preds_petr4 = predictions[predictions["unique_id"] == "PETR4.SA"].set_index("ds")

# Para gerar sinais, precisamos alinhar as previsões com os preços atuais
last_known_price = close["PETR4.SA"].reindex(preds_petr4.index, method="ffill")

# Sinal de compra: se a previsão para o próximo dia (NHITS) for maior que o último preço conhecido.
entries = preds_petr4["NHITS"] > last_known_price
# Sinal de saída: quando a condição de entrada não for mais verdadeira.
exits = ~entries

# 5) Backtest
print("--- Executando Backtest para PETR4.SA ---")
pf = run_backtest(close["PETR4.SA"], entries, exits)
print(pf.stats())
