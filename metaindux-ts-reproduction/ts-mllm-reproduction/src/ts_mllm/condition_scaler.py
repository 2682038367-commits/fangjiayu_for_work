"""Training-only condition-specific Min-Max: a paper-unspecified assumption."""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .data import ACTIVE_SENSOR_COLUMNS, SETTING_COLUMNS


class ConditionMinMaxScaler:
    def __init__(self, n_conditions=6, seed=42):
        self.n_conditions = n_conditions
        self.seed = seed
        self.columns = tuple(ACTIVE_SENSOR_COLUMNS)

    def fit(self, frame):
        settings = frame[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        values = frame[list(self.columns)].to_numpy(dtype=np.float64)
        if len(settings) < self.n_conditions or not np.isfinite(settings).all() or not np.isfinite(values).all():
            raise ValueError("invalid condition-scaler training data")
        self.setting_mean = settings.mean(0)
        std = settings.std(0)
        self.setting_std = np.where(std < 1e-12, 1.0, std)
        scaled = (settings - self.setting_mean) / self.setting_std
        clusters = KMeans(n_clusters=self.n_conditions, random_state=self.seed, n_init=10).fit(scaled)
        raw_centers = clusters.cluster_centers_ * self.setting_std + self.setting_mean
        order = np.lexsort((raw_centers[:, 2], raw_centers[:, 1], raw_centers[:, 0]))
        self.centers = clusters.cluster_centers_[order]
        labels = self.conditions(frame)
        if len(np.unique(labels)) != self.n_conditions:
            raise ValueError("not all condition clusters represented in training")
        self.minimum = np.stack([values[labels == c].min(0) for c in range(self.n_conditions)])
        self.maximum = np.stack([values[labels == c].max(0) for c in range(self.n_conditions)])
        return self

    def conditions(self, frame):
        if not hasattr(self, "centers"):
            raise RuntimeError("condition scaler is not fitted")
        settings = frame[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(settings).all():
            raise ValueError("nonfinite operational settings")
        scaled = (settings - self.setting_mean) / self.setting_std
        return ((scaled[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1).argmin(1)

    def transform(self, frame):
        if not hasattr(self, "minimum"):
            raise RuntimeError("condition scaler is not fitted")
        result = frame.copy()
        labels = self.conditions(frame)
        values = frame[list(self.columns)].to_numpy(dtype=np.float64)
        ranges = self.maximum[labels] - self.minimum[labels]
        normalized = (values - self.minimum[labels]) / np.where(ranges < 1e-12, 1.0, ranges)
        for i, column in enumerate(self.columns):
            result[column] = normalized[:, i].astype(np.float32)
        return result

    def save(self, path):
        record = {"kind": "condition_minmax", "n_conditions": self.n_conditions, "seed": self.seed,
                  "columns": list(self.columns), "setting_columns": SETTING_COLUMNS,
                  "setting_mean": self.setting_mean.tolist(), "setting_std": self.setting_std.tolist(),
                  "centers": self.centers.tolist(), "minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                  "fit_scope": "training engines only; settings standardization, clustering and sensor ranges",
                  "assumption": "KMeans6 on three standardized settings; nearest training centroid at validation/test; no sensor clipping"}
        Path(path).write_text(json.dumps(record, indent=2))

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text())
        if record["kind"] != "condition_minmax":
            raise ValueError("wrong scaler kind")
        scaler = cls(record["n_conditions"], record["seed"])
        scaler.columns = tuple(record["columns"])
        for name in ("setting_mean", "setting_std", "centers", "minimum", "maximum"):
            setattr(scaler, name, np.array(record[name], dtype=np.float64))
        return scaler
