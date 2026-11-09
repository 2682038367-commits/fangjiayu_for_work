import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_mllm.rul_protocol import TARGET_DIVISOR, TARGET_SCALE_STATUS
from ts_mllm.training import predict


def test_b_predictions_restore_cycles_without_scaling_labels():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        model[1].weight.fill_(0.5)
        model[1].bias.zero_()
    loader = DataLoader([{"x": torch.ones(1), "target": torch.tensor(80.0),
                          "unit": torch.tensor(1), "cycle": torch.tensor(40)}])
    predictions, targets, units, cycles = predict(model, loader, torch.device("cpu"), TARGET_DIVISOR)
    np.testing.assert_allclose(predictions, [62.5])
    np.testing.assert_allclose(targets, [80.0])
    np.testing.assert_array_equal(units, [1])
    np.testing.assert_array_equal(cycles, [40])
    native, _, _, _ = predict(model, loader, torch.device("cpu"))
    np.testing.assert_allclose(native, [0.5])
    assert TARGET_SCALE_STATUS == "replication_assumption_paper_unspecified"
