import sys

import pytest

from ts_mllm.pretrain_mae import parse_args as mae_args
from ts_mllm.training_tmaf import parse_args as tmaf_args


@pytest.mark.parametrize("dataset", ["FD002", "FD003", "FD004"])
def test_tmaf_cli_accepts_all_remaining_datasets(monkeypatch, dataset):
    monkeypatch.setattr(sys, "argv", ["train_tmaf", "--dataset", dataset, "--train-stride", "50",
                                      "--validation-stride", "50", "--context-mode", "global_broadcast"])
    args = tmaf_args()
    assert args.dataset == dataset
    assert args.train_stride == args.validation_stride == 50
    assert args.output_bias_init == "default"
    assert args.freeze_temporal_epochs == 0
    assert (args.epochs, args.batch_size, args.learning_rate) == (30, 128, 0.002)


def test_mae_cli_accepts_rebuilt_data_route(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["pretrain_mae", "--dataset", "FD004", "--rebuilt-data-dir", str(tmp_path),
                                      "--device", "cuda"])
    args = mae_args()
    assert args.rebuilt_data_dir == tmp_path
    assert args.device == "cuda"
    assert (args.epochs, args.batch_size, args.learning_rate) == (15, 128, 0.001)
