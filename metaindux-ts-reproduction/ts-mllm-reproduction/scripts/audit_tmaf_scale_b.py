"""Independently replay the scale-B FD001 checkpoint on GPU."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_mllm.qwen_dataset import CachedQwenDataset
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.training_tmaf import predict, require_cuda
from ts_mllm.training import clipped_metrics


def main(output=None):
    root = Path(__file__).resolve().parents[1]
    output = output or root / "artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42"
    result = json.loads((output / "result.json").read_text())
    history = json.loads((output / "history.json").read_text())
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["target_divisor"] == 125.0
    assert checkpoint["temporal_checkpoint_sha256"] == file_sha256(root / result["temporal_checkpoint"])
    assert len(history) == 30
    assert all(not row["temporal_frozen"] and row["temporal_lr"] == row["fusion_lr"] == 0.002 for row in history)
    assert min(history, key=lambda row: row["val_rmse"])["epoch"] == checkpoint["epoch"]
    device = require_cuda()
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast",
        global_pooling=result["model_config"].get("global_pooling", "mean"))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data_dir = Path(result["rebuilt_data_dir"]) if result.get("rebuilt_data_dir") else root / f"artifacts/data_window40_stride50/split_seed{result['split_seed']}/{result['dataset']}"
    assert file_sha256(data_dir / "manifest.json") == result["data_manifest_sha256"]
    train_count = len(RebuiltWindowDataset(data_dir, "train"))
    updates = ((train_count + result["batch_size"] - 1) // result["batch_size"]) * result["epochs"]
    audit = {"device": torch.cuda.get_device_name(0), "target_divisor": 125.0,
             "best_epoch_selected_by_validation": checkpoint["epoch"], "updates": updates}
    for split in ("val", "test"):
        base = RebuiltWindowDataset(data_dir, split)
        dataset = CachedQwenDataset(base, root / result["qwen_cache"], split,
            token_mode=result.get("token_mode", "full"), shuffle_seed=result.get("shuffle_seed") or 42)
        p, y, units, cycles = predict(model, DataLoader(dataset, batch_size=128), device, 125.0)
        metrics = clipped_metrics(y, p, 125)
        expected = result["validation" if split == "val" else "test"]
        for key in metrics:
            np.testing.assert_allclose(metrics[key], expected[key], rtol=1e-5, atol=1e-4)
        assert np.isfinite(p).all()
        if split == "test":
            with (output / "test_predictions.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == len(p) == len(base)
            for name, values in (("prediction_raw", p), ("target", y), ("unit", units), ("cycle", cycles)):
                np.testing.assert_allclose([float(row[name]) for row in rows], values, rtol=1e-5, atol=1e-4)
        audit[split] = {"count": len(p), "metrics": metrics, "raw_prediction_mean": float(p.mean()),
                        "raw_prediction_std": float(p.std()), "finite": True}
    (output / "output_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args().output_dir)
