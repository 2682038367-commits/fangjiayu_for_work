from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from rul_chronos.data import COLUMNS, PrefixDataset, RegimeNormalizer, SENSOR_COLUMNS, add_train_rul, collate_prefixes


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for unit, length in [(1, 3), (2, 2)]:
        for cycle in range(1, length + 1):
            rows.append([unit, cycle, 0.0, 0.0, 0.0, *([float(unit + cycle)] * 21)])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_rul_and_left_padded_prefix_batch():
    frame = add_train_rul(synthetic_frame(), cap=125)
    normalizer = RegimeNormalizer(1).fit(frame)
    dataset = PrefixDataset(normalizer.transform(frame), 1)
    batch = collate_prefixes([dataset[0], dataset[2]])
    assert batch["context"].shape == (2, 21, 3)
    assert torch.isnan(batch["context"][0, :, :2]).all()
    assert batch["target"].tolist() == [2.0, 0.0]

