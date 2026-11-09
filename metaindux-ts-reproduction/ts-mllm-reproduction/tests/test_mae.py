import pytest

torch = pytest.importorskip("torch")

from ts_mllm.mae import (
    SpectrumMaskedAutoencoder,
    SpectrumViTEncoder,
    TemporalViTRegressor,
    VisionTransformerConfig,
)


def test_vit_and_mae_shapes_and_mask_ratio() -> None:
    config = VisionTransformerConfig()
    images = torch.rand(2, 3, 114, 114)
    encoder = SpectrumViTEncoder(config)
    assert config.patch_count == 36
    assert encoder(images).shape == (2, 128)
    mae = SpectrumMaskedAutoencoder(config, mask_ratio=0.75)
    loss, reconstruction, mask = mae(images)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert reconstruction.shape == (2, 36, config.patch_dim)
    assert mask.shape == (2, 36)
    assert mask.sum(dim=1).tolist() == [27.0, 27.0]
    loss.backward()


def test_temporal_vit_regressor_shape() -> None:
    model = TemporalViTRegressor()
    output = model(torch.rand(2, 40, 14), torch.rand(2, 3, 114, 114))
    assert output.shape == (2,)
