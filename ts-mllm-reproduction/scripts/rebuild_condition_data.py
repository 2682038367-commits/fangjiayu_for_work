"""Multi-condition normalization data, retaining A's exact split/window mapping."""
import argparse
import json
from pathlib import Path

import numpy as np

from ts_mllm.condition_scaler import ConditionMinMaxScaler
from ts_mllm.data import CMapssWindowDataset, read_cmapss, add_train_rul, add_test_rul, ACTIVE_SENSOR_COLUMNS
from ts_mllm.rebuilt_data import export_split, file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FD002", "FD004"], default="FD002")
    dataset = parser.parse_args().dataset
    source = root / f"artifacts/data_window40_stride50/split_seed42/{dataset}"
    destination = root / f"artifacts/data_condition_minmax/split_seed42/{dataset}"
    if destination.exists():
        raise FileExistsError("refusing to overwrite condition data")
    manifest = json.loads((source / "manifest.json").read_text())
    for filename, expected in manifest["raw_sha256"].items():
        if file_sha256(root.parent / "data/CMAPSSData" / filename) != expected:
            raise ValueError(f"raw source changed since A export: {filename}")
    raw_train, raw_test, rul = read_cmapss(root.parent / "data/CMAPSSData", dataset)
    fit_frame = raw_train[raw_train.unit.isin(manifest["train_engine_ids"])]
    scaler = ConditionMinMaxScaler().fit(fit_frame)
    train = scaler.transform(add_train_rul(raw_train, 125))
    test = scaler.transform(add_test_rul(raw_test, rul, 125))
    datasets = {"train": CMapssWindowDataset(train, manifest["train_engine_ids"], 40, 50),
                "val": CMapssWindowDataset(train, manifest["val_engine_ids"], 40, 50),
                "test": CMapssWindowDataset(test, window_size=40, endpoints_only=True)}
    destination.mkdir(parents=True)
    splits = {name: export_split(base, destination, name) for name, base in datasets.items()}
    for name in splits:
        for field in ("target", "unit", "cycle", "start_row", "left_padding"):
            np.testing.assert_array_equal(np.load(destination / f"{name}_{field}.npy"), np.load(source / f"{name}_{field}.npy"))
    np.testing.assert_array_equal(np.load(destination / "test_target.npy"), np.minimum(rul, 125))
    scaler.save(destination / "scaler.json")
    manifest.update({"normalization": "condition_minmax", "normalization_assumption": "training-only KMeans6 on standardized three settings; per-condition per-sensor Min-Max",
                     "normalization_reference_manifest_sha256": file_sha256(source / "manifest.json"),
                     "scaler_sha256": file_sha256(destination / "scaler.json"), "splits": splits})
    manifest["assumptions"]["normalization"] = "all scaler and KMeans parameters fit on training engines only; nearest centroid for validation/test; no clipping"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    diagnostics = {"dataset": dataset, "normalization_only_change": "all target/unit/cycle/start_row/left_padding arrays identical to A",
                   "fit_training_engine_count": len(manifest["train_engine_ids"]),
                   "raw_condition_centers": (scaler.centers * scaler.setting_std + scaler.setting_mean).tolist(),
                   "splits": {name: {"count": len(base)} for name, base in datasets.items()}}
    for name, frame in (("train", fit_frame), ("val", raw_train[raw_train.unit.isin(manifest["val_engine_ids"])]), ("test", raw_test)):
        labels = scaler.conditions(frame)
        values = scaler.transform(frame)[ACTIVE_SENSOR_COLUMNS].to_numpy()
        diagnostics[name] = {"condition_counts": np.bincount(labels, minlength=6).tolist(),
                             "outside_0_1_fraction": float(((values < 0) | (values > 1)).mean()),
                             "within_condition_std_median": float(np.median([values[labels == c].std(0) for c in range(6)]))}
    (destination / "normalization_audit.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
