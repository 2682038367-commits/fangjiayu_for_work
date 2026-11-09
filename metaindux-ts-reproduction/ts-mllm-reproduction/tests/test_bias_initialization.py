import pytest
from ts_mllm.bias_initialization import initialize_output_bias
from ts_mllm.tmaf import TemporalMultimodalAttentionFusion


def test_mean_initialization_changes_only_final_bias():
    model = TemporalMultimodalAttentionFusion()
    default = initialize_output_bias(model, "default", 83.466)
    modified = initialize_output_bias(model, "train_mean", 83.466)
    assert modified["changed_state_tensors"] == ["regression_head.3.bias"]
    assert modified["default_initial_state_sha256"] == default["applied_initial_state_sha256"]
    assert modified["unchanged_except_final_bias_sha256"] == default["unchanged_except_final_bias_sha256"]
    assert model.regression_head[-1].bias.item() == pytest.approx(83.466 / 125)


def test_bias_initialization_rejects_unknown_mode():
    with pytest.raises(ValueError):
        initialize_output_bias(TemporalMultimodalAttentionFusion(), "other", 80)
