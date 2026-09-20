#!/usr/bin/env python3
"""Extract resumable FD001 mean-patch caches for pre-registered split seeds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rul_chronos.cache import extract_cache
from rul_chronos.data import prepare_splits
from rul_chronos.encoder import Chronos2Encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
ROOT = PROJECT_ROOT / "artifacts" / "FD001" / "split_sensitivity"
SPLIT_SEEDS = (2026, 2027)
BATCH_SIZE = 24
SPLITS = ("train", "val", "test")


def cache_dir(split_seed: int) -> Path:
    return ROOT / f"split_seed{split_seed}"


def complete(directory: Path, split: str, size: int) -> bool:
    marker = directory / f".{split}.complete"
    metadata_path = directory / f"{split}_metadata.json"
    if not marker.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["size"]) != size or int(metadata["num_regimes"]) != 1:
        return False
    expected = {
        "embeddings": (size, 21, 768),
        "sensors": (size, 21),
        "regimes": (size, 1),
        "targets": (size,),
        "units": (size,),
        "cycles": (size,),
    }
    return all(
        (directory / f"{split}_{key}.npy").exists()
        and np.load(directory / f"{split}_{key}.npy", mmap_mode="r").shape == shape
        for key, shape in expected.items()
    )


def verify(directory: Path, split: str, size: int) -> dict[str, object]:
    arrays = {
        key: np.load(directory / f"{split}_{key}.npy", mmap_mode="r")
        for key in ("embeddings", "sensors", "regimes", "targets", "units", "cycles")
    }
    embeddings = arrays["embeddings"]
    if embeddings.shape != (size, 21, 768) or embeddings.dtype != np.float16:
        raise RuntimeError(f"invalid split_seed cache {split}: {embeddings.shape}, {embeddings.dtype}")
    if any(len(array) != size or not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError(f"non-finite or incomplete cache for {split}")
    return {
        "size": size,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "unique_units": int(len(np.unique(arrays["units"]))),
        "all_finite": True,
    }


def main() -> None:
    prepared = {}
    for split_seed in SPLIT_SEEDS:
        splits, normalizer, train_ids, val_ids = prepare_splits(DATA_DIR, "FD001", seed=split_seed)
        if len(train_ids) != 80 or len(val_ids) != 20 or set(train_ids) & set(val_ids):
            raise RuntimeError(f"invalid engine split for seed {split_seed}")
        if set(train_ids) | set(val_ids) != set(range(1, 101)):
            raise RuntimeError(f"split seed {split_seed} does not cover all 100 training engines")
        directory = cache_dir(split_seed)
        directory.mkdir(parents=True, exist_ok=True)
        normalizer.save(directory / "normalizer.json")
        (directory / "split.json").write_text(
            json.dumps(
                {"seed": split_seed, "train_engine_ids": train_ids, "val_engine_ids": val_ids},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        prepared[split_seed] = (splits, train_ids, val_ids)

    pending = [
        (split_seed, split)
        for split_seed, (splits, _, _) in prepared.items()
        for split in SPLITS
        if not complete(cache_dir(split_seed), split, len(splits[split]))
    ]
    if pending:
        encoder = Chronos2Encoder(
            "amazon/chronos-2", device="cuda", dtype="float16", representation="valid_patch_mean"
        )
        for split_seed, split in pending:
            splits, _, _ = prepared[split_seed]
            directory = cache_dir(split_seed)
            print(f"extracting FD001 split_seed={split_seed} {split}: {len(splits[split])} samples")
            extract_cache(splits[split], encoder, directory, split, batch_size=BATCH_SIZE)
            verify(directory, split, len(splits[split]))
            (directory / f".{split}.complete").write_text("ok\n", encoding="utf-8")

    manifests = {}
    for split_seed, (splits, train_ids, val_ids) in prepared.items():
        directory = cache_dir(split_seed)
        verification = {split: verify(directory, split, len(splits[split])) for split in SPLITS}
        manifest = {
            "dataset": "FD001",
            "split_seed": split_seed,
            "normalizer_fit_engine_ids": train_ids,
            "normalizer_fit_scope": "only the 80 training engines of this split",
            "train_engine_count": len(train_ids),
            "validation_engine_count": len(val_ids),
            "train_validation_overlap": len(set(train_ids) & set(val_ids)),
            "model_id": "amazon/chronos-2",
            "representation": "valid_patch_mean",
            "joint_group_attention": True,
            "group_ids": "torch.arange(batch_size).repeat_interleave(21)",
            "inference_dtype": "float16",
            "cache_dtype": "float16",
            "batch_size": BATCH_SIZE,
            "verification": verification,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[split_seed] = manifest
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "cache_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
