"""Metrics used by the C-MAPSS RUL literature."""

from __future__ import annotations

import numpy as np


def _vectors(targets: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError(f"shape mismatch: targets={targets.shape}, predictions={predictions.shape}")
    if targets.size == 0:
        raise ValueError("metrics require at least one prediction")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("metrics require finite targets and predictions")
    return targets, predictions


def nasa_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """NASA asymmetric score; RUL overestimates receive the stronger penalty."""
    targets, predictions = _vectors(targets, predictions)
    error = predictions - targets
    penalties = np.where(error < 0.0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    return float(penalties.sum())


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets, predictions = _vectors(targets, predictions)
    error = predictions - targets
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "score": nasa_score(targets, predictions),
        "prediction_std": float(np.std(predictions)),
    }
