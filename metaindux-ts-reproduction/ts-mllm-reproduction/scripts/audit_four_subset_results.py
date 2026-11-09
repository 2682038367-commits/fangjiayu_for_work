"""GPU temporal replay and source/CSV checks for all four single-seed results."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_mllm.model import PatchTransformer
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.training import clipped_metrics, predict
from ts_mllm.training_tmaf import require_cuda


def main():
    root = Path(__file__).resolve().parents[1]
    device = require_cuda()
    report = {"device": torch.cuda.get_device_name(0), "stability_not_evaluated": True, "datasets": {}}
    for dataset, expected_count in (("FD001", 100), ("FD002", 259), ("FD003", 100), ("FD004", 248)):
        source = root / f"artifacts/data_window40_stride50/split_seed42/{dataset}"
        temporal_dir = root / ("artifacts/target_scale/FD001/seed42_verified/B_div125" if dataset == "FD001"
                               else f"artifacts/temporal_audited/{dataset}/stride50_B_seed42")
        if dataset == "FD001":
            temporal_result = json.loads((temporal_dir.parent / "summary.json").read_text())["runs"]["B_div125"]
            expected_test = temporal_result["test_cycles"]
            expected_val = temporal_result["validation_cycles"]
        else:
            temporal_result = json.loads((temporal_dir / "result.json").read_text())
            expected_test, expected_val = temporal_result["test"], temporal_result["validation"]
        checkpoint = torch.load(temporal_dir / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["target_divisor"] == 125
        assert checkpoint["data_manifest_sha256"] == file_sha256(source / "manifest.json")
        model = PatchTransformer().to(device)
        model.load_state_dict(checkpoint["model_state"])
        split_report = {}
        for split in ("val", "test"):
            base = RebuiltWindowDataset(source, split)
            p, y, units, cycles = predict(model, DataLoader(base, batch_size=128), device, 125)
            metrics = clipped_metrics(y, p, 125)
            expected = expected_test if split == "test" else expected_val
            for key in metrics:
                np.testing.assert_allclose(metrics[key], expected[key], rtol=1e-5, atol=1e-4)
            if split == "test":
                assert len(p) == expected_count
                with (temporal_dir / "test_predictions.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                assert len(rows) == expected_count
                for field, values in (("target", y), ("unit", units), ("cycle", cycles), ("prediction_raw", p)):
                    np.testing.assert_allclose([float(row[field]) for row in rows], values, rtol=1e-5, atol=1e-4)
            split_report[split] = {"count": len(p), "raw_prediction_std": float(p.std()), "metrics": metrics,
                                   "finite": bool(np.isfinite(p).all()), "low_variance_warning_std_below_1": bool(p.std() < 1)}
        tmaf_dir = root / f"artifacts/tmaf_audited/{dataset}/global_broadcast_stride50_B_seed42"
        tmaf_result = json.loads((tmaf_dir / "result.json").read_text())
        assert tmaf_result["temporal_checkpoint_sha256"] == file_sha256(temporal_dir / "best.pt")
        assert tmaf_result["data_manifest_sha256"] == checkpoint["data_manifest_sha256"]
        assert tmaf_result["target_divisor"] == 125 and tmaf_result["freeze_temporal_epochs"] == 0
        assert tmaf_result["epochs"] == 30 and tmaf_result["batch_size"] == 128
        assert tmaf_result["learning_rate"] == tmaf_result["temporal_learning_rate"] == 0.002
        tmaf_audit = json.loads((tmaf_dir / "output_audit.json").read_text())
        assert tmaf_audit["test"]["count"] == expected_count and tmaf_audit["test"]["finite"]
        if dataset != "FD001":
            mae_dir = root / f"artifacts/mae_pretrain_audited/{dataset}/stride50_seed42"
            alignment = json.loads((root / f"artifacts/svlma_alignment/{dataset}/stride50_seed42/result.json").read_text())
            cache = json.loads((root / f"artifacts/qwen_audited/{dataset}/seed42/manifest.json").read_text())
            assert alignment["vision_checkpoint_sha256"] == cache["vision_checkpoint_sha256"] == file_sha256(mae_dir / "best.pt")
            assert alignment["data_manifest_sha256"] == cache["data_manifest_sha256"] == checkpoint["data_manifest_sha256"]
            assert tmaf_result["initialization"]["mode"] == "default"
        report["datasets"][dataset] = {"temporal": split_report, "tmaf": tmaf_audit,
                                        "checkpoint_and_cache_provenance_checks": "passed"}
        del model
    destination = root / "artifacts/multimodal_remaining_B_seed42/four_subset_audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
