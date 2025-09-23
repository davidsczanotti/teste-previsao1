# src/config.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator
import yaml
import pathlib


class DataCfg(BaseModel):
    tickers: List[str] = Field(..., min_items=1)
    start: str


class ModelCfg(BaseModel):
    horizon: int = 5
    input_size: int = 60
    n_windows: int = 8
    step_size: int = 1
    max_steps: int = 300
    seed: int = 1
    lead_for_signal: int = 1

    @validator("lead_for_signal")
    def _lead_ge1(cls, v):
        if v < 1:
            raise ValueError("lead_for_signal deve ser >= 1")
        return v


class SignalsCfg(BaseModel):
    exp_thresh: float = 0.002
    consec: int = 1
    trend_sma: Optional[int] = None
    dyn_thresh_k: Optional[float] = None
    vol_window: int = 20
    # Filtros opcionais baseados em RSI
    rsi_window: Optional[int] = None
    rsi_min: Optional[float] = None


class BacktestCfg(BaseModel):
    init_cash: float = 100_000.0
    fees: float = 0.0005
    slippage: float = 0.0005
    direction: str = "longonly"
    only_non_overlapping: bool = True
    risk_per_trade: Optional[float] = None  # fração do capital, ex.: 0.005


class TrackingCfg(BaseModel):
    use_mlflow: bool = False
    mlflow_experiment: str = "default"
    mlflow_uri: Optional[str] = None


class ExperimentCfg(BaseModel):
    name: str = "run"
    notes: Optional[str] = None


class Cfg(BaseModel):
    data: DataCfg
    model: ModelCfg
    signals: SignalsCfg
    backtest: BacktestCfg
    tracking: TrackingCfg = TrackingCfg()
    experiment: ExperimentCfg = ExperimentCfg()


def load_config(path: str) -> Cfg:
    p = pathlib.Path(path)
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return Cfg(**raw)
