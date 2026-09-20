import pytest

torch = pytest.importorskip("torch")

from rul_chronos.model import WideDeepRULAdapter


def test_wide_deep_shape_and_backward():
    model = WideDeepRULAdapter(embedding_dim=16, compression_dim=8, num_regimes=6)
    prediction = model(
        torch.randn(4, 21, 16),
        torch.randn(4, 21),
        torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 6).float(),
    )
    assert prediction.shape == (4,)
    prediction.mean().backward()
    assert model.compression.weight.grad is not None


def test_deep_only_does_not_require_wide_inputs():
    model = WideDeepRULAdapter(embedding_dim=16, deep_only=True)
    assert model(torch.randn(2, 21, 16)).shape == (2,)

