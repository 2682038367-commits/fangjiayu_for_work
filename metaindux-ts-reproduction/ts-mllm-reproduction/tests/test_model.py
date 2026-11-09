import pytest

torch = pytest.importorskip("torch")

from ts_mllm.model import PatchTransformer, PatchTransformerConfig


def test_paper_patch_shape_and_prediction_shape() -> None:
    config = PatchTransformerConfig()
    model = PatchTransformer(config)
    inputs = torch.randn(3, 40, 14)
    patches = model.make_patches(inputs)
    assert config.patch_count == 38
    assert patches.shape == (3, 38, 56)
    assert model(inputs).shape == (3,)


def test_model_rejects_wrong_input_shape() -> None:
    model = PatchTransformer()
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(3, 39, 14))
