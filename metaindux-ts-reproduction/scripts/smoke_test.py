#!/usr/bin/env python3
"""Validate data loading, checkpoint loading, and one GPU forward pass."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "upstream"
cache_root = PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("KEOPS_CACHE_FOLDER", str(cache_root / "keops"))
os.chdir(UPSTREAM)
sys.path.insert(0, str(UPSTREAM))

from data.CMAPSSDataset import CMAPSSDataset  # noqa: E402
from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre  # noqa: E402
from GaussianDiffusion import GaussianDiffusion1D_cls_free  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this session")

    device = torch.device("cuda")
    dataset = CMAPSSDataset("FD001", sequence_length=48, deleted_engine=[1000])
    train_frame = dataset.get_train_data()
    features = dataset.get_feature_slice(train_frame)
    labels = dataset.get_label_slice(train_frame)

    model = UNet1D_fre(
        dim=32,
        dim_mults=(1, 2),
        cond_drop_prob=0.2,
        channels=14,
        length=48,
    ).to(device)
    checkpoint = UPSTREAM / "weights" / "DiffUnet_fre_FD001_48.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length=48,
        channels=14,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="linear",
    ).to(device)

    batch = features[:2].permute(0, 2, 1).to(device)
    condition = labels[:2].to(device)
    with torch.no_grad():
        loss = diffusion(batch, classes=condition).sum()

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"train_features={tuple(features.shape)} train_labels={tuple(labels.shape)}")
    print(f"checkpoint={checkpoint.name} forward_loss={loss.item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
