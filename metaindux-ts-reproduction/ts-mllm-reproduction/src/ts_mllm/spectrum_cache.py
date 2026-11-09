"""Precomputed spectrum image datasets for fair visual-encoder comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedSpectrumDataset(Dataset):
    def __init__(self, base: Dataset, image_path: str | Path) -> None:
        self.base = base
        self.image_path = Path(image_path)
        self.images = np.load(self.image_path, mmap_mode="r")
        manifest_path = self.image_path.parent / "manifest.json"
        self.scale = 255.0 if self.images.dtype == np.uint8 else 1.0
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(self.images.dtype) != manifest["dtype"]:
                raise ValueError("spectrum cache dtype does not match manifest")
            self.scale = float(manifest.get("scale", self.scale))
            if self.scale <= 0:
                raise ValueError("spectrum cache scale must be positive")
            if hasattr(base, "data_manifest_sha256"):
                if manifest.get("data_manifest_sha256") != base.data_manifest_sha256:
                    raise ValueError("spectrum cache does not match rebuilt window data")
        if len(self.images) != len(base):
            raise ValueError(
                f"cache length {len(self.images)} does not match dataset length {len(base)}"
            )
        if self.images.shape[1:] != (3, 114, 114):
            raise ValueError(f"unexpected cache shape {self.images.shape}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        # Copy avoids returning a read-only memory-mapped tensor to DataLoader.
        image = np.array(self.images[index], dtype=np.float32, copy=True) / self.scale
        item["image"] = torch.from_numpy(image)
        return item


def load_cache_manifest(cache_dir: str | Path) -> dict[str, object]:
    return json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
