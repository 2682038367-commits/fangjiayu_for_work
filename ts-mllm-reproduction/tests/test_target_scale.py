import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ts_mllm.target_scale import to_training_target, to_cycles, state_hash
from ts_mllm.model import PatchTransformer


def test_target_scaling_preserves_rul_in_original_units():
    labels = torch.tensor([0., 25., 125.])
    normalized = to_training_target(labels, 125)
    torch.testing.assert_close(normalized, torch.tensor([0., 0.2, 1.]))
    np.testing.assert_allclose(to_cycles(normalized.numpy(), 125), labels.numpy())
    np.testing.assert_array_equal(to_cycles(labels.numpy(), 1), labels.numpy())
    with pytest.raises(ValueError):
        to_training_target(labels, 0)


def test_initial_state_hash_detects_parameter_changes():
    model = PatchTransformer()
    first = state_hash(model)
    assert first == state_hash(model)
    with torch.no_grad():
        model.regression_head.bias.add_(1)
    assert first != state_hash(model)
