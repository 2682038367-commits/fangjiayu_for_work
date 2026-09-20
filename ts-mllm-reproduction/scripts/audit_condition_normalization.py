"""GPU A/B replay and independent training-only conditional-data reconstruction."""
import csv
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_mllm.condition_scaler import ConditionMinMaxScaler
from ts_mllm.data import CMapssWindowDataset, read_cmapss, add_train_rul, add_test_rul
from ts_mllm.model import PatchTransformer
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.training import clipped_metrics, predict
from ts_mllm.training_tmaf import require_cuda


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FD002", "FD004"], default="FD002")
    dataset = parser.parse_args().dataset
    a_data = root / f"artifacts/data_window40_stride50/split_seed42/{dataset}"
    b_data = root / f"artifacts/data_condition_minmax/split_seed42/{dataset}"
    a_output = root / f"artifacts/normalization_comparison/{dataset}/global_minmax_seed42"
    b_output = root / f"artifacts/temporal_condition_minmax/{dataset}/stride50_B_seed42"
    a_result, b_result = [json.loads((p / "result.json").read_text()) for p in (a_output, b_output)]
    assert a_result["initial_state_sha256"] == b_result["initial_state_sha256"]
    for key in ("train_config", "model_config", "target_divisor", "target_scale_status"):
        assert a_result[key] == b_result[key], key
    selection = {"selected": "condition_minmax" if b_result["validation"]["rmse"] < a_result["validation"]["rmse"] else "global_minmax",
                 "criterion": "validation clipped RMSE in cycle units", "test_not_used_for_selection": True,
                 "initial_state_sha256_identical": a_result["initial_state_sha256"],
                 "A_validation_rmse": a_result["validation"]["rmse"], "B_validation_rmse": b_result["validation"]["rmse"]}
    (b_output / "selection.json").write_text(json.dumps(selection, indent=2))
    manifest = json.loads((b_data / "manifest.json").read_text())
    for filename, expected in manifest["raw_sha256"].items():
        assert file_sha256(root.parent / "data/CMAPSSData" / filename) == expected
    scaler = ConditionMinMaxScaler.load(b_data / "scaler.json")
    raw_train, raw_test, rul = read_cmapss(root.parent / "data/CMAPSSData", dataset)
    np.testing.assert_array_equal(np.load(b_data / "test_unit.npy"), np.sort(raw_test.unit.unique()))
    np.testing.assert_array_equal(np.load(b_data / "test_target.npy"), np.minimum(rul, 125))
    refitted = ConditionMinMaxScaler().fit(raw_train[raw_train.unit.isin(manifest["train_engine_ids"])])
    for name in ("minimum", "maximum", "centers", "setting_mean", "setting_std"):
        np.testing.assert_allclose(getattr(scaler, name), getattr(refitted, name), atol=1e-10, rtol=1e-10)
    train = scaler.transform(add_train_rul(raw_train))
    test = scaler.transform(add_test_rul(raw_test, rul))
    bases = {"train": CMapssWindowDataset(train, manifest["train_engine_ids"], 40, 50),
             "val": CMapssWindowDataset(train, manifest["val_engine_ids"], 40, 50),
             "test": CMapssWindowDataset(test, window_size=40, endpoints_only=True)}
    for split, base in bases.items():
        saved = RebuiltWindowDataset(b_data, split)
        for i in range(len(base)):
            for field in ("x", "target", "unit", "cycle"):
                torch.testing.assert_close(base[i][field], saved[i][field], rtol=0, atol=0)
        for field in ("target", "unit", "cycle", "start_row", "left_padding"):
            np.testing.assert_array_equal(np.load(b_data / f"{split}_{field}.npy"), np.load(a_data / f"{split}_{field}.npy"))
    report = {"dataset": dataset, "selection": selection, "training_only_refit_matches_saved_scaler": True,
              "independent_reconstruction_all_rows": "passed", "labels_and_window_mapping_identical": True, "runs": {}}
    device = require_cuda()
    report["device"] = torch.cuda.get_device_name(0)
    for name, data, output, result in (("A_global", a_data, a_output, a_result), ("B_condition", b_data, b_output, b_result)):
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["data_manifest_sha256"] == file_sha256(data / "manifest.json")
        assert file_sha256(output / "scaler.json") == file_sha256(data / "scaler.json")
        assert checkpoint["target_divisor"] == 125
        history = json.loads((output / "history.json").read_text())
        assert len(history) == 30 and min(history, key=lambda row: row["val_rmse"])["epoch"] == checkpoint["epoch"]
        model = PatchTransformer().to(device)
        model.load_state_dict(checkpoint["model_state"])
        train_count = len(RebuiltWindowDataset(data, "train"))
        updates = ((train_count + result["train_config"]["batch_size"] - 1) // result["train_config"]["batch_size"]) * len(history)
        report["runs"][name] = {"best_epoch": checkpoint["epoch"], "updates": updates}
        for split in ("val", "test"):
            base = RebuiltWindowDataset(data, split)
            p, y, units, cycles = predict(model, DataLoader(base, batch_size=128), device, 125)
            metrics = clipped_metrics(y, p, 125)
            for key in metrics:
                np.testing.assert_allclose(metrics[key], result["validation" if split == "val" else "test"][key], rtol=1e-5, atol=1e-4)
            assert np.isfinite(p).all()
            if split == "test":
                with (output / "test_predictions.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                assert len(rows) == len(p) == len(rul)
                for field, values in (("prediction_raw", p), ("target", y), ("unit", units), ("cycle", cycles)):
                    np.testing.assert_allclose([float(row[field]) for row in rows], values, rtol=1e-5, atol=1e-4)
            report["runs"][name][split] = {"count": len(p), "metrics": metrics, "raw_prediction_mean": float(p.mean()),
                                           "raw_prediction_std": float(p.std()), "prediction_target_correlation": float(np.corrcoef(p, y)[0, 1])}
        del model
    (b_output / "output_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
