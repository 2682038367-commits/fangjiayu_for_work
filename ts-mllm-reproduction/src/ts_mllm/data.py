"""Leakage-safe C-MAPSS windowing for the TS-MLLM reproduction.

The paper removes seven sensor channels, caps RUL at 125, and uses windows
of 40 cycles.  Its stated sliding-window stride of 50 is retained as a
configurable option, while the default is the conventional stride of 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SETTING_COLUMNS = [f"setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]
REMOVED_SENSOR_INDICES = (1, 5, 6, 10, 16, 18, 19)
ACTIVE_SENSOR_COLUMNS = [
    column
    for index, column in enumerate(SENSOR_COLUMNS, start=1)
    if index not in REMOVED_SENSOR_INDICES
]
VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}


def _validate_dataset(dataset: str) -> str:
    dataset = dataset.upper()
    if dataset not in VALID_DATASETS:
        expected = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {expected}")
    return dataset


def read_cmapss(
    data_dir: str | Path,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Read one official C-MAPSS subset and validate its engine/RUL mapping."""
    dataset = _validate_dataset(dataset)
    data_dir = Path(data_dir)
    read_kwargs = {"sep": r"\s+", "header": None, "names": COLUMNS, "engine": "python"}
    train = pd.read_csv(data_dir / f"train_{dataset}.txt", **read_kwargs)
    test = pd.read_csv(data_dir / f"test_{dataset}.txt", **read_kwargs)
    test_rul = pd.read_csv(
        data_dir / f"RUL_{dataset}.txt", sep=r"\s+", header=None, engine="python"
    ).iloc[:, 0].to_numpy(dtype=np.float32)

    test_units = np.sort(test["unit"].unique())
    if len(test_units) != len(test_rul):
        raise ValueError(
            f"{dataset}: {len(test_units)} test engines but {len(test_rul)} RUL labels"
        )
    return train, test, test_rul


@dataclass
class TrainMinMaxScaler:
    """Per-sensor min-max scaler fitted on training engines only."""

    columns: tuple[str, ...] = tuple(ACTIVE_SENSOR_COLUMNS)
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "TrainMinMaxScaler":
        values = frame[list(self.columns)].to_numpy(dtype=np.float64)
        if values.size == 0:
            raise ValueError("cannot fit the scaler on an empty frame")
        self.minimum = values.min(axis=0)
        self.maximum = values.max(axis=0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = frame.copy()
        values = result[list(self.columns)].to_numpy(dtype=np.float64)
        scale = self.maximum - self.minimum  # type: ignore[operator]
        safe_scale = np.where(scale < 1e-12, 1.0, scale)
        normalized = ((values - self.minimum) / safe_scale).astype(np.float32)
        # A constant training channel is represented as zero. Values outside the
        # training range are intentionally not clipped, so distribution shift is visible.
        for index, column in enumerate(self.columns):
            result[column] = normalized[:, index]
        return result

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        payload = {
            "columns": list(self.columns),
            "minimum": self.minimum.tolist(),  # type: ignore[union-attr]
            "maximum": self.maximum.tolist(),  # type: ignore[union-attr]
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainMinMaxScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            columns=tuple(payload["columns"]),
            minimum=np.asarray(payload["minimum"], dtype=np.float64),
            maximum=np.asarray(payload["maximum"], dtype=np.float64),
        )

    def _check_fitted(self) -> None:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("TrainMinMaxScaler has not been fitted")


def add_train_rul(frame: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Add piecewise-linear train labels: min(cap, final_cycle - cycle)."""
    result = frame.copy()
    final_cycle = result.groupby("unit")["cycle"].transform("max")
    result["rul"] = np.minimum(cap, final_cycle - result["cycle"]).astype(np.float32)
    return result


def add_test_rul(frame: pd.DataFrame, endpoint_rul: np.ndarray, cap: int = 125) -> pd.DataFrame:
    """Add labels to observed test cycles from the official endpoint RUL file."""
    result = frame.copy()
    units = np.sort(result["unit"].unique())
    if len(units) != len(endpoint_rul):
        raise ValueError("test engine count does not match endpoint RUL count")
    endpoint_by_unit = dict(zip(units.tolist(), endpoint_rul.tolist()))
    final_cycle = result.groupby("unit")["cycle"].transform("max")
    residual = result["unit"].map(endpoint_by_unit).to_numpy(dtype=np.float32)
    result["rul"] = np.minimum(cap, residual + final_cycle - result["cycle"]).astype(np.float32)
    return result


def split_engine_ids(
    frame: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.8,
) -> tuple[list[int], list[int]]:
    """Deterministically split whole engines, never individual windows."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    units = np.sort(frame["unit"].unique()).astype(np.int64)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    count = int(round(len(units) * train_fraction))
    count = min(max(count, 1), len(units) - 1)
    return sorted(shuffled[:count].tolist()), sorted(shuffled[count:].tolist())


def select_few_shot_engines(
    engine_ids: Sequence[int],
    fraction: float,
    seed: int,
) -> list[int]:
    """Select a reproducible fraction of engines for few-shot experiments."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("few_shot_fraction must be in (0, 1]")
    ids = np.asarray(sorted(engine_ids), dtype=np.int64)
    if fraction == 1.0:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    count = max(1, int(round(len(ids) * fraction)))
    return sorted(rng.choice(ids, size=count, replace=False).tolist())


class CMapssWindowDataset(Dataset):
    """Fixed windows shaped ``[window_size, 14]`` with endpoint RUL labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        engine_ids: Sequence[int] | None = None,
        window_size: int = 40,
        stride: int = 1,
        endpoints_only: bool = False,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if engine_ids is not None:
            frame = frame[frame["unit"].isin(engine_ids)]
        self.window_size = window_size
        self.stride = stride
        self.endpoints_only = endpoints_only
        self.engines = {
            int(unit): trajectory.sort_values("cycle").reset_index(drop=True)
            for unit, trajectory in frame.groupby("unit", sort=True)
        }
        self.index: list[tuple[int, int]] = []
        for unit, trajectory in self.engines.items():
            if endpoints_only:
                # Official test trajectories can be shorter than the paper's
                # 40-cycle window. Keep every engine and left-pad such cases in
                # __getitem__ instead of silently losing its official RUL label.
                starts = [max(0, len(trajectory) - window_size)]
            else:
                if len(trajectory) < window_size:
                    continue
                starts = range(0, len(trajectory) - window_size + 1, stride)
            self.index.extend((unit, int(start)) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        unit, start = self.index[index]
        window = self.engines[unit].iloc[start : start + self.window_size]
        last = window.iloc[-1]
        values = window[ACTIVE_SENSOR_COLUMNS].to_numpy(dtype=np.float32).copy()
        if len(values) < self.window_size:
            missing = self.window_size - len(values)
            values = np.concatenate(
                [np.repeat(values[:1], missing, axis=0), values], axis=0
            )
        return {
            "x": torch.from_numpy(values),
            "target": torch.tensor(float(last["rul"]), dtype=torch.float32),
            "unit": torch.tensor(unit, dtype=torch.int64),
            "cycle": torch.tensor(int(last["cycle"]), dtype=torch.int64),
        }


@dataclass
class CMapssDataBundle:
    dataset: str
    train: CMapssWindowDataset
    val: CMapssWindowDataset
    test: CMapssWindowDataset
    scaler: TrainMinMaxScaler
    train_engine_ids: list[int]
    val_engine_ids: list[int]
    official_train_engine_ids: list[int]


def prepare_cmapss_data(
    data_dir: str | Path,
    dataset: str,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    few_shot_fraction: float = 1.0,
    window_size: int = 40,
    train_stride: int = 1,
    validation_stride: int = 1,
    rul_cap: int = 125,
) -> CMapssDataBundle:
    """Build leakage-safe train/validation/test datasets for one subset.

    The scaler is fitted after engine-level splitting and few-shot selection,
    using only engines that actually participate in training.
    """
    dataset = _validate_dataset(dataset)
    raw_train, raw_test, endpoint_rul = read_cmapss(data_dir, dataset)
    official_train_ids, val_ids = split_engine_ids(
        raw_train, seed=seed, train_fraction=1.0 - validation_fraction
    )
    selected_train_ids = select_few_shot_engines(
        official_train_ids, fraction=few_shot_fraction, seed=seed
    )

    scaler = TrainMinMaxScaler().fit(raw_train[raw_train["unit"].isin(selected_train_ids)])
    train = scaler.transform(add_train_rul(raw_train, cap=rul_cap))
    test = scaler.transform(add_test_rul(raw_test, endpoint_rul, cap=rul_cap))

    return CMapssDataBundle(
        dataset=dataset,
        train=CMapssWindowDataset(
            train,
            selected_train_ids,
            window_size=window_size,
            stride=train_stride,
        ),
        val=CMapssWindowDataset(
            train,
            val_ids,
            window_size=window_size,
            stride=validation_stride,
        ),
        test=CMapssWindowDataset(
            test,
            window_size=window_size,
            stride=1,
            endpoints_only=True,
        ),
        scaler=scaler,
        train_engine_ids=selected_train_ids,
        val_engine_ids=val_ids,
        official_train_engine_ids=official_train_ids,
    )
