import torch
from torch import nn

from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion


def test_head_preserves_explicit_table_ii_values():
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast"))
    assert model.regression_head[0].out_features == 512
    assert isinstance(model.regression_head[2], nn.Dropout)
    assert model.regression_head[2].p == 0.5


def test_default_transformer_clones_have_equal_values_without_shared_storage():
    model = TemporalMultimodalAttentionFusion()
    first, second = model.temporal.encoder.layers
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
