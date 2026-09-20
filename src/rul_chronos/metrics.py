from __future__ import annotations

import numpy as np


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction - target))))


def phm_score(target: np.ndarray, prediction: np.ndarray) -> float:
    """Official asymmetric C-MAPSS score (late predictions cost more)."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    penalties = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(penalties.sum())

