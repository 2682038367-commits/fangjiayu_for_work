#!/usr/bin/env python3
"""Extract FD003 Chronos-2 REG and future-query representations in one forward pass."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from rul_chronos.data import collate_prefixes, prepare_splits
from rul_chronos.encoder import Chronos2Encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
BASE_CACHE = PROJECT_ROOT / "artifacts" / "FD003"
OUTPUT_ROOT = BASE_CACHE / "token_representation_ablation"
REPRESENTATIONS = ("reg_token", "future_query")
BATCH_SIZE = 24


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    os.link(source, destination)


def prepare_directories() -> None:
    for representation in REPRESENTATIONS:
        directory = OUTPUT_ROOT / representation
        directory.mkdir(parents=True, exist_ok=True)
        hardlink(BASE_CACHE / "normalizer.json", directory / "normalizer.json")
        hardlink(BASE_CACHE / "split.json", directory / "split.json")
        for split in ("train", "val", "test"):
            hardlink(BASE_CACHE / f"{split}_metadata.json", directory / f"{split}_metadata.json")
            for key in ("sensors", "regimes", "targets", "units", "cycles"):
                hardlink(BASE_CACHE / f"{split}_{key}.npy", directory / f"{split}_{key}.npy")


def main() -> None:
    splits, _, train_ids, val_ids = prepare_splits(DATA_DIR, "FD003", seed=42)
    expected_split = json.loads((BASE_CACHE / "split.json").read_text(encoding="utf-8"))
    if train_ids != expected_split["train_engine_ids"] or val_ids != expected_split["val_engine_ids"]:
        raise RuntimeError("reconstructed engine split differs from the existing FD003 cache")
    prepare_directories()
    encoder = Chronos2Encoder("amazon/chronos-2", device="cuda", dtype="float16")
    if not encoder.model.chronos_config.use_reg_token:
        raise RuntimeError("amazon/chronos-2 does not expose the expected REG token")

    for split, dataset in splits.items():
        metadata = json.loads((BASE_CACHE / f"{split}_metadata.json").read_text(encoding="utf-8"))
        size = int(metadata["size"])
        arrays = {
            representation: np.lib.format.open_memmap(
                OUTPUT_ROOT / representation / f"{split}_embeddings.npy",
                mode="w+",
                dtype=np.float16,
                shape=(size, int(metadata["num_sensors"]), int(metadata["embedding_dim"])),
            )
            for representation in REPRESENTATIONS
        }
        base_units = np.load(BASE_CACHE / f"{split}_units.npy", mmap_mode="r")
        base_cycles = np.load(BASE_CACHE / f"{split}_cycles.npy", mmap_mode="r")
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_prefixes)
        offset = 0
        for batch in tqdm(loader, desc=f"FD003 special tokens {split}", unit="batch"):
            variants = encoder.embed_variants(batch["context"], REPRESENTATIONS)
            count = len(batch["unit"])
            region = slice(offset, offset + count)
            if not np.array_equal(batch["unit"].numpy(), base_units[region]):
                raise RuntimeError(f"{split} unit order differs at offset {offset}")
            if not np.array_equal(batch["cycle"].numpy(), base_cycles[region]):
                raise RuntimeError(f"{split} cycle order differs at offset {offset}")
            for representation in REPRESENTATIONS:
                arrays[representation][region] = variants[representation].numpy().astype(np.float16)
            offset += count
        if offset != size:
            raise RuntimeError(f"{split} extraction wrote {offset} rows, expected {size}")
        for array in arrays.values():
            array.flush()

    manifest = {
        "dataset": "FD003",
        "seed": 42,
        "model_id": "amazon/chronos-2",
        "inference_dtype": "float16",
        "cache_dtype": "float16",
        "batch_size": BATCH_SIZE,
        "num_output_patches": 1,
        "encoder_sequence": "context patches -> REG -> one future query",
        "group_ids": "torch.arange(batch_size).repeat_interleave(21)",
        "representations": {
            "reg_token": "encoder hidden state at num_context_patches",
            "future_query": "encoder hidden state at num_context_patches + 1",
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"token representation caches written to {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
