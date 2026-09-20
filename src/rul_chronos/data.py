from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
COMPLEX_DATASETS = {"FD002", "FD004"}


def _dataset_name(name: str) -> str:
    value = name.upper()
    if value not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"unknown dataset {name!r}; expected FD001, FD002, FD003, or FD004")
    return value


def read_cmapss(data_dir: str | Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    data_dir = Path(data_dir)
    dataset = _dataset_name(dataset)
    kwargs = dict(sep=r"\s+", header=None, names=COLUMNS, engine="python")
    train = pd.read_csv(data_dir / f"train_{dataset}.txt", **kwargs)
    test = pd.read_csv(data_dir / f"test_{dataset}.txt", **kwargs)
    test_rul = pd.read_csv(data_dir / f"RUL_{dataset}.txt", sep=r"\s+", header=None, engine="python").iloc[:, 0]
    if test["unit"].nunique() != len(test_rul):
        raise ValueError("number of test engines does not match the RUL file")
    return train, test, test_rul.to_numpy(dtype=np.float32)


@dataclass
class RegimeNormalizer:
    num_regimes: int
    centers: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, seed: int = 42) -> "RegimeNormalizer":
        settings = frame[SETTING_COLUMNS].to_numpy(dtype=np.float64)
        if self.num_regimes == 1:
            labels = np.zeros(len(frame), dtype=np.int64)
            self.centers = settings.mean(axis=0, keepdims=True)
        else:
            kmeans = KMeans(n_clusters=self.num_regimes, random_state=seed, n_init=20)
            labels = kmeans.fit_predict(settings)
            self.centers = kmeans.cluster_centers_
        sensors = frame[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        self.means = np.empty((self.num_regimes, len(SENSOR_COLUMNS)), dtype=np.float64)
        self.stds = np.empty_like(self.means)
        for regime in range(self.num_regimes):
            selected = sensors[labels == regime]
            if len(selected) == 0:
                raise ValueError(f"regime {regime} contains no training samples")
            self.means[regime] = selected.mean(axis=0)
            std = selected.std(axis=0, ddof=0)
            self.stds[regime] = np.where(std < 1e-8, 1.0, std)
        return self

    def predict_regime(self, settings: np.ndarray) -> np.ndarray:
        self._check_fitted()
        settings = np.asarray(settings, dtype=np.float64)
        distances = np.square(settings[:, None, :] - self.centers[None, :, :]).sum(axis=-1)
        return distances.argmin(axis=1).astype(np.int64)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = frame.copy()
        labels = self.predict_regime(result[SETTING_COLUMNS].to_numpy())
        sensors = result[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
        normalized = (sensors - self.means[labels]) / self.stds[labels]
        normalized = normalized.astype(np.float32)
        # Assign columns individually so integer-valued sensors (for example the
        # constant FD001 channels) are replaced with float columns instead of
        # triggering pandas' incompatible-dtype assignment path.
        for index, column in enumerate(SENSOR_COLUMNS):
            result[column] = normalized[:, index]
        result["regime"] = labels
        return result

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        payload = {
            "num_regimes": self.num_regimes,
            "centers": self.centers.tolist(),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RegimeNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            num_regimes=int(payload["num_regimes"]),
            centers=np.asarray(payload["centers"], dtype=np.float64),
            means=np.asarray(payload["means"], dtype=np.float64),
            stds=np.asarray(payload["stds"], dtype=np.float64),
        )

    def _check_fitted(self) -> None:
        if self.centers is None or self.means is None or self.stds is None:
            raise RuntimeError("RegimeNormalizer has not been fitted")


def split_engine_ids(frame: pd.DataFrame, seed: int = 42) -> tuple[list[int], list[int]]:
    groups = frame["unit"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=groups))
    return sorted(frame.iloc[train_idx]["unit"].unique().tolist()), sorted(frame.iloc[val_idx]["unit"].unique().tolist())


def add_train_rul(frame: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    result = frame.copy()
    final_cycles = result.groupby("unit")["cycle"].transform("max")
    result["rul"] = np.minimum(cap, final_cycles - result["cycle"]).astype(np.float32)
    return result


class PrefixDataset(Dataset):
    """Each item is the complete observed history ending at one engine cycle."""

    def __init__(
        self,
        frame: pd.DataFrame,
        num_regimes: int,
        engine_ids: Sequence[int] | None = None,
        endpoints_only: bool = False,
    ) -> None:
        if engine_ids is not None:
            frame = frame[frame["unit"].isin(engine_ids)]
        self.num_regimes = num_regimes
        self.engines: dict[int, pd.DataFrame] = {
            int(unit): group.sort_values("cycle").reset_index(drop=True)
            for unit, group in frame.groupby("unit", sort=True)
        }
        self.index: list[tuple[int, int]] = []
        for unit, trajectory in self.engines.items():
            endpoints = [len(trajectory) - 1] if endpoints_only else range(len(trajectory))
            self.index.extend((unit, int(endpoint)) for endpoint in endpoints)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, object]:
        unit, endpoint = self.index[index]
        trajectory = self.engines[unit]
        history = trajectory.iloc[: endpoint + 1]
        last = history.iloc[-1]
        regime = int(last["regime"])
        return {
            "context": torch.from_numpy(history[SENSOR_COLUMNS].to_numpy(dtype=np.float32).T.copy()),
            "sensors": torch.from_numpy(last[SENSOR_COLUMNS].to_numpy(dtype=np.float32).copy()),
            "regimes": torch.nn.functional.one_hot(torch.tensor(regime), self.num_regimes).to(torch.float32),
            "target": torch.tensor(float(last["rul"]), dtype=torch.float32),
            "unit": unit,
            "cycle": int(last["cycle"]),
        }


def collate_prefixes(items: Sequence[dict[str, object]]) -> dict[str, torch.Tensor]:
    max_length = max(item["context"].shape[-1] for item in items)  # type: ignore[union-attr]
    contexts = torch.full((len(items), len(SENSOR_COLUMNS), max_length), torch.nan, dtype=torch.float32)
    for row, item in enumerate(items):
        context = item["context"]
        contexts[row, :, -context.shape[-1] :] = context  # type: ignore[union-attr]
    return {
        "context": contexts,
        "sensors": torch.stack([item["sensors"] for item in items]),  # type: ignore[list-item]
        "regimes": torch.stack([item["regimes"] for item in items]),  # type: ignore[list-item]
        "target": torch.stack([item["target"] for item in items]),  # type: ignore[list-item]
        "unit": torch.tensor([item["unit"] for item in items], dtype=torch.int64),
        "cycle": torch.tensor([item["cycle"] for item in items], dtype=torch.int64),
    }


def prepare_splits(
    data_dir: str | Path,
    dataset: str,
    seed: int = 42,
    rul_cap: int = 125,
) -> tuple[dict[str, PrefixDataset], RegimeNormalizer, list[int], list[int]]:
    dataset = _dataset_name(dataset)
    train, test, test_rul = read_cmapss(data_dir, dataset)
    train_ids, val_ids = split_engine_ids(train, seed)
    fit_frame = train[train["unit"].isin(train_ids)]
    normalizer = RegimeNormalizer(6 if dataset in COMPLEX_DATASETS else 1).fit(fit_frame, seed)
    train = normalizer.transform(add_train_rul(train, rul_cap))
    test = normalizer.transform(test)
    # RUL files are ordered by engine ID and label the final observed test cycle.
    test_rul_by_unit = dict(zip(sorted(test["unit"].unique()), test_rul.tolist()))
    test["rul"] = np.minimum(
        rul_cap,
        test["unit"].map(test_rul_by_unit).to_numpy(dtype=np.float32),
    ).astype(np.float32)
    splits = {
        "train": PrefixDataset(train, normalizer.num_regimes, train_ids),
        "val": PrefixDataset(train, normalizer.num_regimes, val_ids),
        "test": PrefixDataset(test, normalizer.num_regimes, endpoints_only=True),
    }
    return splits, normalizer, train_ids, val_ids
