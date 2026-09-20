"""Durable, separately versioned window40/sample-stride50 data artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import ACTIVE_SENSOR_COLUMNS, CMapssWindowDataset, prepare_cmapss_data, read_cmapss
from .model import PatchTransformerConfig


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_split(base: CMapssWindowDataset, output_dir: Path, name: str) -> dict[str, object]:
    rows = [base[index] for index in range(len(base))]
    arrays = {
        "x": np.stack([item["x"].numpy() for item in rows]).astype(np.float32),
        "target": np.array([float(item["target"]) for item in rows], dtype=np.float32),
        "unit": np.array([int(item["unit"]) for item in rows], dtype=np.int64),
        "cycle": np.array([int(item["cycle"]) for item in rows], dtype=np.int64),
        "start_row": np.array([start for _, start in base.index], dtype=np.int64),
        "left_padding": np.array([
            max(0, base.window_size - len(base.engines[unit]))
            for unit, _ in base.index
        ], dtype=np.int64),
    }
    if arrays["x"].shape != (len(base), 40, 14):
        raise ValueError(f"{name}: wrong window shape")
    if not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["target"]).all():
        raise ValueError(f"{name}: non-finite values")
    if not ((arrays["target"] >= 0) & (arrays["target"] <= 125)).all():
        raise ValueError(f"{name}: RUL outside [0,125]")
    for unit, trajectory in base.engines.items():
        indices = np.flatnonzero(arrays["unit"] == unit)
        if base.endpoints_only:
            assert len(indices) == 1
            assert int(arrays["cycle"][indices[0]]) == int(trajectory["cycle"].max())
        else:
            expected = max(0, (len(trajectory) - 40) // 50 + 1)
            assert len(indices) == expected
            np.testing.assert_array_equal(arrays["start_row"][indices], np.arange(expected) * 50)
            assert np.all(np.diff(arrays["target"][indices]) <= 0)
        for index in indices:
            start = int(arrays["start_row"][index])
            last = trajectory.iloc[min(start + 40, len(trajectory)) - 1]
            assert arrays["target"][index] == float(last["rul"])
            assert arrays["cycle"][index] == int(last["cycle"])
    files = {}
    for field, values in arrays.items():
        path = output_dir / f"{name}_{field}.npy"
        np.save(path, values, allow_pickle=False)
        loaded = np.load(path, allow_pickle=False)
        np.testing.assert_array_equal(loaded, values)
        files[field] = {"file": path.name, "sha256": file_sha256(path)}
    return {
        "count": len(base),
        "shape": list(arrays["x"].shape),
        "x_dtype": "float32",
        "engine_count": len(np.unique(arrays["unit"])),
        "left_padded_windows": int((arrays["left_padding"] > 0).sum()),
        "files": files,
        "audit": "passed: shape, finite, RUL, count, start spacing, endpoint labels, disk roundtrip",
    }


def rebuild_dataset(data_dir: Path, output_dir: Path, dataset: str, seed: int = 42) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite data artifact: {output_dir}")
    bundle = prepare_cmapss_data(
        data_dir, dataset, seed=seed, window_size=40,
        train_stride=50, validation_stride=50, rul_cap=125,
    )
    assert set(bundle.train_engine_ids).isdisjoint(bundle.val_engine_ids)
    patch = PatchTransformerConfig()
    assert patch.patch_stride == 1 and patch.patch_size == 4 and patch.patch_count == 38
    output_dir.mkdir(parents=True)
    bundle.scaler.save(output_dir / "scaler.json")
    splits = {
        name: export_split(base, output_dir, name)
        for name, base in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test))
    }
    _, raw_test, official_rul = read_cmapss(data_dir, dataset)
    units = np.load(output_dir / "test_unit.npy")
    targets = np.load(output_dir / "test_target.npy")
    np.testing.assert_array_equal(units, np.sort(raw_test.unit.unique()))
    np.testing.assert_array_equal(targets, np.minimum(official_rul, 125))
    manifest = {
        "dataset": dataset,
        "protocol": "window40_sample_stride50_v1",
        "fidelity_status": "paper_explicit_window_stride_with_disclosed_data_protocol_assumptions",
        "split_seed": seed,
        "window_size": 40,
        "train_sample_stride": 50,
        "validation_sample_stride": 50,
        "patch_size": 4,
        "patch_stride": 1,
        "patch_count": 38,
        "rul_cap": 125,
        "sensor_columns": ACTIVE_SENSOR_COLUMNS,
        "train_engine_ids": bundle.train_engine_ids,
        "val_engine_ids": bundle.val_engine_ids,
        "assumptions": {
            "validation": "engine-level 80/20 split; apply sample stride50 to validation too",
            "normalization": "fit on training engines only; no clipping of unseen ranges",
            "windowing": "start at row0; keep full stride50 windows; do not append off-stride final window",
            "label": "RUL at window endpoint, capped at125, including test labels",
            "test": "one last endpoint window per engine; left-pad short trajectories by first observation",
        },
        "raw_sha256": {
            f"{prefix}_{dataset}.txt": file_sha256(data_dir / f"{prefix}_{dataset}.txt")
            for prefix in ("train", "test", "RUL")
        },
        "scaler_sha256": file_sha256(output_dir / "scaler.json"),
        "splits": splits,
        "official_test_rul_mapping": "passed (capped at125)",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class RebuiltWindowDataset(Dataset):
    """Use the exported data order directly in future spectrum/cache stages."""

    def __init__(self, data_dir: str | Path, split: str, verify_hashes: bool = True) -> None:
        data_dir = Path(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.data_manifest_sha256 = file_sha256(data_dir / "manifest.json")
        if manifest["protocol"] != "window40_sample_stride50_v1":
            raise ValueError("unexpected rebuilt data protocol")
        self.arrays = {}
        for field in ("x", "target", "unit", "cycle"):
            descriptor = manifest["splits"][split]["files"][field]
            path = data_dir / descriptor["file"]
            if verify_hashes and file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"checksum mismatch: {path}")
            self.arrays[field] = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.arrays["x"].shape != (manifest["splits"][split]["count"], 40, 14):
            raise ValueError("unexpected exported window shape")

    def __len__(self) -> int:
        return len(self.arrays["x"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            field: torch.from_numpy(np.array(values[index], copy=True))
            for field, values in self.arrays.items()
        }
