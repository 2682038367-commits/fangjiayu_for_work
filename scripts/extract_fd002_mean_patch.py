#!/usr/bin/env python3
"""Extract a resumable FD002 valid-patch-mean Chronos-2 cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rul_chronos.cache import extract_cache
from rul_chronos.data import prepare_splits
from rul_chronos.encoder import Chronos2Encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD002"
SEED = 42
BATCH_SIZE = 24
SPLIT_NAMES = ("train", "val", "test")


def cache_is_complete(split: str, expected_size: int) -> bool:
    marker = CACHE_DIR / f".{split}.complete"
    metadata_path = CACHE_DIR / f"{split}_metadata.json"
    if not marker.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shapes = {
        "embeddings": (expected_size, 21, 768),
        "sensors": (expected_size, 21),
        "regimes": (expected_size, 6),
        "targets": (expected_size,),
        "units": (expected_size,),
        "cycles": (expected_size,),
    }
    if int(metadata["size"]) != expected_size:
        return False
    for key, shape in expected_shapes.items():
        path = CACHE_DIR / f"{split}_{key}.npy"
        if not path.exists() or np.load(path, mmap_mode="r").shape != shape:
            return False
    return True


def verify_split(split: str, expected_size: int) -> dict[str, object]:
    arrays = {
        key: np.load(CACHE_DIR / f"{split}_{key}.npy", mmap_mode="r")
        for key in ("embeddings", "sensors", "regimes", "targets", "units", "cycles")
    }
    if arrays["embeddings"].shape != (expected_size, 21, 768):
        raise RuntimeError(f"unexpected {split} embedding shape: {arrays['embeddings'].shape}")
    if arrays["embeddings"].dtype != np.float16:
        raise RuntimeError(f"unexpected {split} embedding dtype: {arrays['embeddings'].dtype}")
    for key, array in arrays.items():
        if len(array) != expected_size or not np.isfinite(array).all():
            raise RuntimeError(f"invalid {split} {key} cache")
    unique_units = np.unique(arrays["units"])
    return {
        "size": expected_size,
        "embedding_shape": list(arrays["embeddings"].shape),
        "embedding_dtype": str(arrays["embeddings"].dtype),
        "unique_units": int(len(unique_units)),
        "all_finite": True,
    }


def main() -> None:
    splits, normalizer, train_ids, val_ids = prepare_splits(DATA_DIR, "FD002", seed=SEED)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    normalizer.save(CACHE_DIR / "normalizer.json")
    (CACHE_DIR / "split.json").write_text(
        json.dumps(
            {"seed": SEED, "train_engine_ids": train_ids, "val_engine_ids": val_ids},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pending = [name for name in SPLIT_NAMES if not cache_is_complete(name, len(splits[name]))]
    if pending:
        encoder = Chronos2Encoder(
            "amazon/chronos-2",
            device="cuda",
            dtype="float16",
            representation="valid_patch_mean",
        )
        for split in pending:
            print(f"extracting FD002 {split}: {len(splits[split])} samples")
            extract_cache(splits[split], encoder, CACHE_DIR, split, batch_size=BATCH_SIZE)
            verify_split(split, len(splits[split]))
            (CACHE_DIR / f".{split}.complete").write_text("ok\n", encoding="utf-8")
    else:
        print("all FD002 cache splits already complete; skipping extraction")

    verification = {name: verify_split(name, len(splits[name])) for name in SPLIT_NAMES}
    manifest = {
        "dataset": "FD002",
        "seed": SEED,
        "model_id": "amazon/chronos-2",
        "representation": "valid_patch_mean",
        "representation_definition": (
            "mean of Chronos-2 encoder context hidden states over patches containing at least one observed value"
        ),
        "joint_group_attention": True,
        "group_ids": "torch.arange(batch_size).repeat_interleave(21)",
        "inference_dtype": "float16",
        "cache_dtype": "float16",
        "batch_size": BATCH_SIZE,
        "num_regimes": 6,
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "train_engines": len(train_ids),
        "validation_engines": len(val_ids),
        "verification": verification,
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
