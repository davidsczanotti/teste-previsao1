# src/models_ts.py (trecho)
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATS, PatchTST


def build_neuralforecast(models, freq="D"):
    return NeuralForecast(models=models, freq=freq)


def nhits_cfg():
    return NHITS(
        h=5,
        input_size=60,
        max_steps=200,
        n_blocks=[1, 1, 1],
        num_layers=[2, 2, 2],
        mlp_units=[[512, 512], [512, 512], [512, 512]],
    )


def nbeats_cfg():
    return NBEATS(
        h=5,
        input_size=60,
        max_steps=200,
        stack_types=["trend", "seasonality"],
        n_blocks=[3, 3],
        num_layers=[4, 4],
        mlp_units=[[256, 256], [256, 256]],
    )


def patchtst_cfg():
    return PatchTST(h=5, input_size=96, max_steps=200, n_layers=3)


def make_tft_dataset(df, group="ticker", time="date", target="ret_fwd"):
    # df precisa estar no formato "long": date, ticker, target, covariates...
    training = TimeSeriesDataSet(
        df,
        time_idx=time,
        target=target,
        group_ids=[group],
        max_encoder_length=60,
        max_prediction_length=5,
        time_varying_known_reals=[],
        time_varying_unknown_reals=[target, "rsi_14", "MACD_12_26_9"],
        allow_missing_timesteps=True,
    )
    return training


def build_tft(training):
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        loss=torch.nn.L1Loss(),  # ou QuantileLoss p/ previsões por quantis
        dropout=0.1,
    )
    return model
