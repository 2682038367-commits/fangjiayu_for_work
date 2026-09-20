"""Dataset adapter joining C-MAPSS windows with precomputed Qwen tokens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


VALID_TOKEN_MODES = {"full", "visual_only", "text_only", "shuffled"}


class CachedQwenDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        cache_dir: str | Path,
        split: str,
        *,
        token_mode: str = "full",
        shuffle_seed: int = 42,
    ) -> None:
        if token_mode not in VALID_TOKEN_MODES:
            raise ValueError(f"unknown token mode {token_mode!r}")
        self.base = base
        self.token_mode = token_mode
        cache_dir = Path(cache_dir)
        self.tokens = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
        self.mask = np.load(cache_dir / f"{split}_mask.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / f"{split}_target.npy", mmap_mode="r")
        self.units = np.load(cache_dir / f"{split}_unit.npy", mmap_mode="r")
        self.cycles = np.load(cache_dir / f"{split}_cycle.npy", mmap_mode="r")
        if (self.tokens.ndim != 3 or self.tokens.shape[0] != len(base)
                or self.tokens.shape[1] < 2 or self.tokens.shape[2] != 1024):
            raise ValueError(
                f"{split}: expected cache [{len(base)},T>=2,1024], got {self.tokens.shape}"
            )
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError(f"{split}: mask shape does not match token cache")
        for values in (self.targets, self.units, self.cycles):
            if values.shape != (len(base),):
                raise ValueError(f"{split}: cached metadata length mismatch")
        self._validate_alignment(split)
        self.token_indices = np.arange(len(base), dtype=np.int64)
        if token_mode == "shuffled":
            if len(base) < 2:
                raise ValueError("shuffled token ablation needs at least two samples")
            # Construct a seeded derangement: randomize the row order and rotate
            # token sources within that order, guaranteeing zero fixed points.
            order = np.random.default_rng(shuffle_seed).permutation(len(base))
            self.token_indices[order] = np.roll(order, 1)

    def _validate_alignment(self, split: str) -> None:
        # Validate every row. This is inexpensive relative to training and makes
        # accidental reuse of a cache from another split/seed fail immediately.
        for index in range(len(self.base)):
            item = self.base[index]
            if int(item["unit"]) != int(self.units[index]):
                raise ValueError(f"{split}: unit mismatch at row {index}")
            if int(item["cycle"]) != int(self.cycles[index]):
                raise ValueError(f"{split}: cycle mismatch at row {index}")
            if not np.isclose(float(item["target"]), float(self.targets[index])):
                raise ValueError(f"{split}: target mismatch at row {index}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[index])
        token_index = int(self.token_indices[index])
        item["llm_tokens"] = torch.from_numpy(
            np.array(self.tokens[token_index], dtype=np.float32, copy=True)
        )
        mask = np.array(self.mask[token_index], dtype=np.bool_, copy=True)
        if self.token_mode == "visual_only":
            mask[1:] = False
        elif self.token_mode == "text_only":
            mask[0] = False
        item["llm_mask"] = torch.from_numpy(mask)
        return item
