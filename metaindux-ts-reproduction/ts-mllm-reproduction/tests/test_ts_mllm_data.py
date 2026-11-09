from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ts_mllm.data import (
    ACTIVE_SENSOR_COLUMNS,
    COLUMNS,
    CMapssWindowDataset,
    TrainMinMaxScaler,
    add_test_rul,
    add_train_rul,
    select_few_shot_engines,
    split_engine_ids,
)


def synthetic_frame(lengths: tuple[int, ...] = (45, 42, 41, 40, 43)) -> pd.DataFrame:
    rows = []
    for unit, length in enumerate(lengths, start=1):
        for cycle in range(1, length + 1):
            settings = [unit * 0.1, 0.0, 1.0]
            sensors = [unit * 100.0 + cycle + index for index in range(1, 22)]
            rows.append([unit, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_active_sensor_selection_matches_paper() -> None:
    assert ACTIVE_SENSOR_COLUMNS == [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ]


def test_train_rul_is_capped_and_non_increasing() -> None:
    frame = add_train_rul(synthetic_frame((140,)), cap=125)
    assert frame.iloc[0]["rul"] == 125
    assert frame.iloc[-1]["rul"] == 0
    assert np.all(np.diff(frame["rul"].to_numpy()) <= 0)


def test_test_endpoint_matches_official_rul() -> None:
    frame = add_test_rul(synthetic_frame((45, 42)), np.asarray([17, 130]), cap=125)
    endpoints = frame.groupby("unit").tail(1)["rul"].tolist()
    assert endpoints == [17.0, 125.0]


def test_scaler_uses_fit_frame_only_and_round_trips(tmp_path: Path) -> None:
    train = synthetic_frame((40,))
    shifted = synthetic_frame((40,)).assign(
        **{column: lambda data, column=column: data[column] + 10_000 for column in ACTIVE_SENSOR_COLUMNS}
    )
    scaler = TrainMinMaxScaler().fit(train)
    transformed = scaler.transform(shifted)
    assert transformed[ACTIVE_SENSOR_COLUMNS].to_numpy().max() > 1.0
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = TrainMinMaxScaler.load(path)
    np.testing.assert_allclose(loaded.transform(train)[ACTIVE_SENSOR_COLUMNS], scaler.transform(train)[ACTIVE_SENSOR_COLUMNS])


def test_window_shape_label_and_endpoint_behavior() -> None:
    frame = add_train_rul(synthetic_frame((45,)))
    frame = TrainMinMaxScaler().fit(frame).transform(frame)
    windows = CMapssWindowDataset(frame, window_size=40, stride=1)
    assert len(windows) == 6
    assert tuple(windows[0]["x"].shape) == (40, 14)
    assert float(windows[0]["target"]) == 5.0
    endpoint = CMapssWindowDataset(frame, window_size=40, endpoints_only=True)
    assert len(endpoint) == 1
    assert int(endpoint[0]["cycle"]) == 45
    assert float(endpoint[0]["target"]) == 0.0


def test_short_test_trajectory_is_left_padded_and_retained() -> None:
    frame = add_train_rul(synthetic_frame((12,)))
    frame = TrainMinMaxScaler().fit(frame).transform(frame)
    endpoint = CMapssWindowDataset(frame, window_size=40, endpoints_only=True)
    assert len(endpoint) == 1
    assert tuple(endpoint[0]["x"].shape) == (40, 14)
    assert int(endpoint[0]["cycle"]) == 12
    torch.testing.assert_close(endpoint[0]["x"][0], endpoint[0]["x"][28])
    assert not torch.equal(endpoint[0]["x"][28], endpoint[0]["x"][29])


def test_engine_split_and_few_shot_are_reproducible_and_disjoint() -> None:
    frame = synthetic_frame()
    train_a, val_a = split_engine_ids(frame, seed=7, train_fraction=0.8)
    train_b, val_b = split_engine_ids(frame, seed=7, train_fraction=0.8)
    assert (train_a, val_a) == (train_b, val_b)
    assert set(train_a).isdisjoint(val_a)
    selected = select_few_shot_engines(train_a, fraction=0.5, seed=7)
    assert set(selected).issubset(train_a)
    assert len(selected) == 2


def test_paper_sample_stride50_does_not_change_patch_stride(tmp_path) -> None:
    from ts_mllm.rebuilt_data import export_split
    from ts_mllm.model import PatchTransformerConfig
    frame = add_train_rul(synthetic_frame((141,)))
    frame = TrainMinMaxScaler().fit(frame).transform(frame)
    windows = CMapssWindowDataset(frame, window_size=40, stride=50)
    assert windows.index == [(1, 0), (1, 50), (1, 100)]
    assert [int(windows[i]["cycle"]) for i in range(len(windows))] == [40, 90, 140]
    assert [float(windows[i]["target"]) for i in range(len(windows))] == [101, 51, 1]
    report = export_split(windows, tmp_path, "train")
    assert report["count"] == 3
    config = PatchTransformerConfig()
    assert config.patch_stride == 1
    assert config.patch_count == 38
