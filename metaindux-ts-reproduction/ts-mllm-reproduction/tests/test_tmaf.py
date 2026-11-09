import pytest
import numpy as np

torch = pytest.importorskip("torch")

from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.qwen_dataset import CachedQwenDataset
from ts_mllm.training_tmaf import set_temporal_phase


def test_tmaf_shapes_and_masked_attention() -> None:
    model = TemporalMultimodalAttentionFusion(TMAFConfig())
    inputs = torch.rand(3, 40, 14)
    tokens = torch.rand(3, 32, 1024)
    mask = torch.ones(3, 32, dtype=torch.bool)
    mask[:, -4:] = False
    fused, attention = model.fuse(inputs, tokens, mask)
    prediction = model(inputs, tokens, mask)
    assert fused.shape == (3, 38, 64)
    assert attention.shape == (3, 38, 32)
    assert prediction.shape == (3,)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(3, 38))
    assert torch.count_nonzero(attention[..., -4:]) == 0


def test_tmaf_rejects_invalid_llm_shape() -> None:
    model = TemporalMultimodalAttentionFusion()
    with pytest.raises(ValueError, match="expected LLM tokens"):
        model(torch.rand(2, 40, 14), torch.rand(2, 32, 100))


def test_temporal_freeze_and_unfreeze() -> None:
    model = TemporalMultimodalAttentionFusion()
    model.train()
    assert set_temporal_phase(model, 5, 5)
    assert not model.temporal.training
    assert not any(p.requires_grad for p in model.temporal.parameters())
    prediction = model(torch.rand(2, 40, 14), torch.rand(2, 32, 1024))
    prediction.sum().backward()
    assert all(p.grad is None for p in model.temporal.parameters())
    assert model.query_projection.weight.grad is not None
    model.train()
    assert not set_temporal_phase(model, 6, 5)
    assert model.temporal.training
    assert model.temporal.patch_projection.weight.requires_grad
    assert not model.temporal.regression_head.weight.requires_grad


def test_literal_global_broadcast_attention_is_uniform() -> None:
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast"))
    fused, attention = model.fuse(torch.rand(2, 40, 14), torch.rand(2, 513, 1024))
    assert fused.shape == (2, 38, 64)
    torch.testing.assert_close(attention, torch.full((2, 38, 38), 1 / 38))


def test_last_text_uses_last_valid_position_not_padding_or_prefix():
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast", global_pooling="last_text")).eval()
    x = torch.rand(2,40,14)
    t = torch.rand(2,6,1024)
    mask = torch.tensor([[True,True,False,True,False,False], [True,True,True,False,False,False]])
    reference = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast")).eval()
    reference.load_state_dict(model.state_dict())
    expected = torch.stack([t[0,3], t[1,2]]).unsqueeze(1)
    with torch.no_grad():
        fused,a = model.fuse(x,t,mask)
        ref,_ = reference.fuse(x,expected)
        torch.testing.assert_close(fused,ref)
        torch.testing.assert_close(a,torch.full((2,38,38),1/38))
        changed = t.clone(); changed[:,0] = 1000; changed[~mask] = -1000
        torch.testing.assert_close(model(x,t,mask),model(x,changed,mask))


def test_last_text_rejects_visual_only_input():
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast",global_pooling="last_text"))
    with pytest.raises(ValueError,match="valid text token"):
        model(torch.rand(1,40,14),torch.rand(1,3,1024),torch.tensor([[True,False,False]]))


class _TinyBase:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.zeros(40, 14),
            "target": torch.tensor(float(index)),
            "unit": torch.tensor(index + 1),
            "cycle": torch.tensor(index + 40),
        }


def _write_tiny_cache(path) -> None:
    np.save(path / "train.npy", np.stack([np.full((32, 1024), i) for i in range(4)]).astype(np.float16))
    np.save(path / "train_mask.npy", np.ones((4, 32), dtype=np.bool_))
    np.save(path / "train_target.npy", np.arange(4, dtype=np.float32))
    np.save(path / "train_unit.npy", np.arange(1, 5, dtype=np.int64))
    np.save(path / "train_cycle.npy", np.arange(40, 44, dtype=np.int64))


def test_cached_qwen_modality_masks_and_derangement(tmp_path) -> None:
    _write_tiny_cache(tmp_path)
    base = _TinyBase()
    visual = CachedQwenDataset(base, tmp_path, "train", token_mode="visual_only")
    text = CachedQwenDataset(base, tmp_path, "train", token_mode="text_only")
    shuffled = CachedQwenDataset(base, tmp_path, "train", token_mode="shuffled", shuffle_seed=7)
    assert visual[0]["llm_mask"].tolist() == [True] + [False] * 31
    assert text[0]["llm_mask"].tolist() == [False] + [True] * 31
    assert np.all(shuffled.token_indices != np.arange(4))
    for index in range(4):
        assert float(shuffled[index]["llm_tokens"][0, 0]) != index
