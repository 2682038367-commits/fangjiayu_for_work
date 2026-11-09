import pytest

torch = pytest.importorskip("torch")

from ts_mllm.vision import SpectrumCNNEncoder, SpectrumTransform, TemporalVisualRegressor


def test_spectrum_transform_shape_range_and_gradients() -> None:
    transform = SpectrumTransform()
    inputs = torch.rand(2, 40, 14)
    images = transform(inputs)
    assert images.shape == (2, 3, 114, 114)
    assert torch.isfinite(images).all()
    assert float(images.detach().min()) >= 0.0
    assert float(images.detach().max()) <= 1.0
    images.mean().backward()
    assert transform.stft_sensor_logits.grad is not None
    assert transform.cwt_sensor_logits.grad is not None


def test_visual_encoder_and_fused_regressor_shapes() -> None:
    images = torch.rand(2, 3, 114, 114)
    assert SpectrumCNNEncoder()(images).shape == (2, 128)
    model = TemporalVisualRegressor()
    assert model(torch.rand(2, 40, 14)).shape == (2,)
