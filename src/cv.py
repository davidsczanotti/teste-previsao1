# src/cv.py
from sklearn.metrics import mean_absolute_error
import numpy as np
from vectorbt.base.resampling import resample_apply


def purged_cv(X, y, sample_weight=None, n_splits=5, embargo_td=None):
    # VectorBT não tem um PurgedKFold direto, mas podemos simular o walk-forward com embargo
    # Para simplificar, vamos usar um walk-forward simples por enquanto e integrar o embargo depois
    scores = []
    # Implementação de PurgedKFold com embargo é mais complexa e será feita em um passo futuro
    # Por enquanto, vamos retornar um score dummy para que o código compile
    return 0.0, []
