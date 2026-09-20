import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from rul_chronos.cache import EmbeddingCacheDataset, extract_cache
from rul_chronos.data import COLUMNS, PrefixDataset, RegimeNormalizer, add_train_rul
from rul_chronos.encoder import DeterministicMockEncoder
from rul_chronos.training import _dataloader_random_order, _finish_dataloader_random_order


def test_extract_cache_round_trip(tmp_path):
    import pandas as pd

    rows = [
        [1, cycle, 0.0, 0.0, 0.0, *([float(cycle)] * 21)]
        for cycle in range(1, 4)
    ]
    frame = add_train_rul(pd.DataFrame(rows, columns=COLUMNS))
    normalizer = RegimeNormalizer(1).fit(frame)
    dataset = PrefixDataset(normalizer.transform(frame), num_regimes=1)

    extract_cache(dataset, DeterministicMockEncoder(embedding_dim=6), tmp_path, "train", batch_size=2)
    cached = EmbeddingCacheDataset(tmp_path, "train")

    assert len(cached) == 3
    assert cached[0]["embeddings"].shape == (21, 6)
    assert [float(cached[i]["target"]) for i in range(3)] == [2.0, 1.0, 0.0]
    assert [int(cached[i]["unit"]) for i in range(3)] == [1, 1, 1]
    assert [int(cached[i]["cycle"]) for i in range(3)] == [1, 2, 3]

    filtered = EmbeddingCacheDataset(tmp_path, "train", min_prefix_length=2)
    assert len(filtered) == 2
    assert filtered.metadata["unfiltered_size"] == 3
    assert [int(filtered[i]["cycle"]) for i in range(2)] == [2, 3]
    assert [float(filtered[i]["target"]) for i in range(2)] == [1.0, 0.0]


def test_cached_gpu_order_matches_dataloader_random_sampler():
    size = 17
    batch_size = 5
    seed = 123
    expected_generator = torch.Generator().manual_seed(seed)
    actual_generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.arange(size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=expected_generator,
        num_workers=0,
    )

    for _ in range(3):
        expected = torch.cat([batch[0] for batch in loader])
        actual = _dataloader_random_order(size, actual_generator)
        _finish_dataloader_random_order(size, actual_generator)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
