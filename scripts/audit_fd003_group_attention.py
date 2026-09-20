#!/usr/bin/env python3
"""Numerically identify whether cached embeddings used joint or independent Chronos grouping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rul_chronos.data import collate_prefixes, prepare_splits
from rul_chronos.encoder import Chronos2Encoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
SAMPLE_INDICES = (31, 127)


@torch.inference_mode()
def independent_variants(encoder: Chronos2Encoder, context: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, num_sensors, history_length = context.shape
    model_context_length = int(encoder.model.chronos_config.context_length)
    if history_length > model_context_length:
        context = context[..., -model_context_length:]
    flat = context.reshape(batch_size * num_sensors, -1).to(encoder.device, dtype=torch.float32)
    context_mask = torch.isfinite(flat).to(encoder.model.dtype)
    patched_mask = torch.nan_to_num(encoder.model.patch(context_mask), nan=0.0)
    valid_patch_mask = patched_mask.sum(dim=-1) > 0
    outputs, _, _, num_context_patches = encoder.model.encode(
        context=flat,
        group_ids=None,
        num_output_patches=1,
    )
    hidden = outputs.last_hidden_state[:, :num_context_patches]
    positions = torch.arange(num_context_patches, device=encoder.device).expand_as(valid_patch_mask)
    last_indices = positions.masked_fill(~valid_patch_mask, -1).max(dim=-1).values
    row_indices = torch.arange(len(hidden), device=encoder.device)
    last = hidden[row_indices, last_indices]
    weights = valid_patch_mask.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return {
        "last_valid_patch": last.reshape(batch_size, num_sensors, encoder.embedding_dim).float().cpu(),
        "valid_patch_mean": mean.reshape(batch_size, num_sensors, encoder.embedding_dim).float().cpu(),
    }


def difference(reference: np.ndarray, candidate: torch.Tensor) -> dict[str, float]:
    delta = reference.astype(np.float32) - candidate.numpy()
    return {
        "mean_abs": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("FD001", "FD003"), default="FD003")
    dataset_name = parser.parse_args().dataset
    base_cache = PROJECT_ROOT / "artifacts" / dataset_name
    representation_directory = "representation_ablation" if dataset_name == "FD001" else "patch_representation_ablation"
    representation_root = base_cache / representation_directory
    mean_cache = representation_root / "valid_patch_mean"
    output_path = representation_root / "GROUP_ATTENTION_AUDIT.json"

    splits, _, _, _ = prepare_splits(DATA_DIR, dataset_name, seed=42)
    loader = DataLoader(
        Subset(splits["train"], list(SAMPLE_INDICES)),
        batch_size=len(SAMPLE_INDICES),
        shuffle=False,
        collate_fn=collate_prefixes,
    )
    batch = next(iter(loader))
    cached_units = np.load(base_cache / "train_units.npy", mmap_mode="r")[list(SAMPLE_INDICES)]
    cached_cycles = np.load(base_cache / "train_cycles.npy", mmap_mode="r")[list(SAMPLE_INDICES)]
    if not np.array_equal(batch["unit"].numpy(), cached_units):
        raise RuntimeError("sample units do not align with the FD003 cache")
    if not np.array_equal(batch["cycle"].numpy(), cached_cycles):
        raise RuntimeError("sample cycles do not align with the FD003 cache")

    encoder = Chronos2Encoder("amazon/chronos-2", device="cuda", dtype="float16")
    joint = encoder.embed_variants(batch["context"], ("last_valid_patch", "valid_patch_mean"))
    independent = independent_variants(encoder, batch["context"])
    cache_paths = {
        "last_valid_patch": base_cache / "train_embeddings.npy",
        "valid_patch_mean": mean_cache / "train_embeddings.npy",
    }
    comparisons = {}
    for representation, path in cache_paths.items():
        cached = np.load(path, mmap_mode="r")[list(SAMPLE_INDICES)]
        comparisons[representation] = {
            "cache_vs_explicit_joint": difference(cached, joint[representation]),
            "cache_vs_default_independent": difference(cached, independent[representation]),
        }
        if comparisons[representation]["cache_vs_explicit_joint"]["max_abs"] > 0.02:
            raise RuntimeError(f"{representation} cache does not match explicit joint grouping")
        if comparisons[representation]["cache_vs_default_independent"]["mean_abs"] < 0.01:
            raise RuntimeError(f"{representation} audit cannot distinguish independent grouping")

    result = {
        "dataset": dataset_name,
        "sample_indices": list(SAMPLE_INDICES),
        "units": cached_units.astype(int).tolist(),
        "cycles": cached_cycles.astype(int).tolist(),
        "batch_size": len(SAMPLE_INDICES),
        "num_sensors": 21,
        "explicit_group_ids": ([0] * 21) + ([1] * 21),
        "default_independent_group_ids": list(range(42)),
        "comparisons": comparisons,
        "conclusion": f"both existing {dataset_name} caches match explicit per-prefix joint grouping, not default independent grouping",
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
