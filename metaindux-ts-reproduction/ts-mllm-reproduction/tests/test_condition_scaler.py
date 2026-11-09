import numpy as np
import pandas as pd
import pytest

from ts_mllm.condition_scaler import ConditionMinMaxScaler
from ts_mllm.data import ACTIVE_SENSOR_COLUMNS, SETTING_COLUMNS


def training_frame():
    rows = []
    for condition in (0, 1):
        for cycle in range(6):
            row = {"unit": condition + 1, "cycle": cycle + 1,
                   SETTING_COLUMNS[0]: condition * 10.0,
                   SETTING_COLUMNS[1]: condition * 0.5,
                   SETTING_COLUMNS[2]: 100.0}
            row.update({column: condition * 100 + cycle for column in ACTIVE_SENSOR_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_condition_minmax_roundtrip_and_metadata_preservation(tmp_path):
    frame = training_frame()
    scaler = ConditionMinMaxScaler(n_conditions=2).fit(frame)
    transformed = scaler.transform(frame)
    for column in ACTIVE_SENSOR_COLUMNS:
        np.testing.assert_allclose(transformed[column], np.tile(np.linspace(0, 1, 6), 2), atol=1e-7)
    pd.testing.assert_frame_equal(frame[["unit", "cycle", *SETTING_COLUMNS]], transformed[["unit", "cycle", *SETTING_COLUMNS]])
    scaler.save(tmp_path / "scaler.json")
    loaded = ConditionMinMaxScaler.load(tmp_path / "scaler.json")
    pd.testing.assert_frame_equal(transformed, loaded.transform(frame))


def test_validation_uses_training_centroids_ranges_and_does_not_clip():
    frame = training_frame()
    scaler = ConditionMinMaxScaler(n_conditions=2).fit(frame)
    minima, maxima, centers = scaler.minimum.copy(), scaler.maximum.copy(), scaler.centers.copy()
    validation = frame.iloc[[0]].copy()
    validation[ACTIVE_SENSOR_COLUMNS] = 20.0
    result = scaler.transform(validation)
    np.testing.assert_allclose(result[ACTIVE_SENSOR_COLUMNS], 4.0)
    np.testing.assert_array_equal(scaler.minimum, minima)
    np.testing.assert_array_equal(scaler.maximum, maxima)
    np.testing.assert_array_equal(scaler.centers, centers)


def test_unfitted_condition_scaler_is_rejected():
    with pytest.raises(RuntimeError):
        ConditionMinMaxScaler().transform(training_frame())
