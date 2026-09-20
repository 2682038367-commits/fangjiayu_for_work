from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import PrefixDataset, collate_prefixes
from .encoder import Encoder


def extract_cache(
    dataset: PrefixDataset,
    encoder: Encoder,
    output_dir: str | Path,
    split: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    size = len(dataset)
    num_sensors = 21
    embedding_dim = encoder.embedding_dim
    arrays = {
        "embeddings": np.lib.format.open_memmap(
            output_dir / f"{split}_embeddings.npy", mode="w+", dtype=np.float16, shape=(size, num_sensors, embedding_dim)
        ),
        "sensors": np.lib.format.open_memmap(
            output_dir / f"{split}_sensors.npy", mode="w+", dtype=np.float32, shape=(size, num_sensors)
        ),
        "regimes": np.lib.format.open_memmap(
            output_dir / f"{split}_regimes.npy", mode="w+", dtype=np.float32, shape=(size, dataset.num_regimes)
        ),
        "targets": np.lib.format.open_memmap(output_dir / f"{split}_targets.npy", mode="w+", dtype=np.float32, shape=(size,)),
        "units": np.lib.format.open_memmap(output_dir / f"{split}_units.npy", mode="w+", dtype=np.int32, shape=(size,)),
        "cycles": np.lib.format.open_memmap(output_dir / f"{split}_cycles.npy", mode="w+", dtype=np.int32, shape=(size,)),
    }
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_prefixes,
        pin_memory=torch.cuda.is_available(),
    )
    offset = 0
    for batch in tqdm(loader, desc=f"extract {split}", unit="batch"):
        embeddings = encoder.embed(batch["context"])
        count = len(embeddings)
        region = slice(offset, offset + count)
        arrays["embeddings"][region] = embeddings.numpy().astype(np.float16)
        batch_keys = {
            "sensors": "sensors",
            "regimes": "regimes",
            "targets": "target",
            "units": "unit",
            "cycles": "cycle",
        }
        for array_key, batch_key in batch_keys.items():
            arrays[array_key][region] = batch[batch_key].numpy()
        offset += count
    for array in arrays.values():
        array.flush()
    metadata = {
        "size": size,
        "num_sensors": num_sensors,
        "embedding_dim": embedding_dim,
        "num_regimes": dataset.num_regimes,
    }
    (output_dir / f"{split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class EmbeddingCacheDataset(Dataset):
    def __init__(self, directory: str | Path, split: str, min_prefix_length: int = 1) -> None:
        if min_prefix_length < 1:
            raise ValueError("min_prefix_length must be at least 1")
        self.directory = Path(directory)
        self.split = split
        self.metadata = json.loads((self.directory / f"{split}_metadata.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(self.directory / f"{split}_embeddings.npy", mmap_mode="r")
        self.sensors = np.load(self.directory / f"{split}_sensors.npy", mmap_mode="r")
        self.regimes = np.load(self.directory / f"{split}_regimes.npy", mmap_mode="r")
        self.targets = np.load(self.directory / f"{split}_targets.npy", mmap_mode="r")
        self.units = np.load(self.directory / f"{split}_units.npy", mmap_mode="r")
        self.cycles = np.load(self.directory / f"{split}_cycles.npy", mmap_mode="r")
        self.min_prefix_length = int(min_prefix_length)
        self.indices = (
            None
            if self.min_prefix_length == 1
            else np.flatnonzero(self.cycles >= self.min_prefix_length).astype(np.int64, copy=False)
        )
        filtered_size = len(self.cycles) if self.indices is None else len(self.indices)
        self.metadata = {
            **self.metadata,
            "unfiltered_size": int(self.metadata["size"]),
            "size": int(filtered_size),
            "min_prefix_length": self.min_prefix_length,
        }

    def __len__(self) -> int:
        return int(self.metadata["size"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = index if self.indices is None else int(self.indices[index])
        # np.array copies read-only memmap slices and avoids PyTorch's non-writable warning.
        return {
            "embeddings": torch.from_numpy(np.array(self.embeddings[source_index], dtype=np.float32)),
            "sensors": torch.from_numpy(np.array(self.sensors[source_index], dtype=np.float32)),
            "regimes": torch.from_numpy(np.array(self.regimes[source_index], dtype=np.float32)),
            "target": torch.tensor(float(self.targets[source_index]), dtype=torch.float32),
            "unit": torch.tensor(int(self.units[source_index]), dtype=torch.int64),
            "cycle": torch.tensor(int(self.cycles[source_index]), dtype=torch.int64),
        }
