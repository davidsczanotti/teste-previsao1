# src/cv.py
from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import pandas as pd


def purged_kfold_split(
    ds: pd.Series,  # index temporal (datetime) da amostra
    n_splits: int = 5,
    embargo: int = 5,  # barras de embargo entre folds
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Retorna gerador de (idx_treino, idx_teste) com purge + embargo."""
    n = len(ds)
    indices = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((start, stop))
        current = stop

    for i, (start, stop) in enumerate(folds):
        test_idx = indices[start:stop]
        # purge vizinhos + embargo
        left = max(0, start - embargo)
        right = min(n, stop + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[left:right] = False
        yield indices[train_mask], test_idx
