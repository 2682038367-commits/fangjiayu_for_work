#!/usr/bin/env python3
"""Audit the TS-MLLM C-MAPSS data pipeline without training a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ts_mllm.data import ACTIVE_SENSOR_COLUMNS, prepare_cmapss_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DATA = PROJECT_ROOT.parent / "data" / "CMAPSSData"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_DATA)
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--few-shot-fraction", type=float, default=1.0)
    return parser.parse_args()


def audit(dataset: str, args: argparse.Namespace) -> None:
    bundle = prepare_cmapss_data(
        args.data_dir,
        dataset,
        seed=args.seed,
        window_size=args.window_size,
        train_stride=args.train_stride,
        few_shot_fraction=args.few_shot_fraction,
    )
    first = bundle.train[0]
    if tuple(first["x"].shape) != (args.window_size, len(ACTIVE_SENSOR_COLUMNS)):
        raise AssertionError(f"unexpected input shape: {tuple(first['x'].shape)}")

    train_units = set(bundle.train_engine_ids)
    val_units = set(bundle.val_engine_ids)
    if train_units & val_units:
        raise AssertionError("training and validation engines overlap")

    for split_name, split in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        targets_by_unit: dict[int, list[tuple[int, float]]] = {}
        for index in range(len(split)):
            item = split[index]
            unit = int(item["unit"])
            cycle = int(item["cycle"])
            target = float(item["target"])
            if not 0.0 <= target <= 125.0:
                raise AssertionError(f"{split_name}: RUL outside [0, 125]")
            targets_by_unit.setdefault(unit, []).append((cycle, target))
        for observations in targets_by_unit.values():
            observations.sort()
            targets = np.asarray([target for _, target in observations])
            if np.any(np.diff(targets) > 1e-6):
                raise AssertionError(f"{split_name}: RUL is not non-increasing")

    print(
        f"{dataset}: PASS | sensors={len(ACTIVE_SENSOR_COLUMNS)} "
        f"shape={tuple(first['x'].shape)} engines(train/val/test)="
        f"{len(bundle.train_engine_ids)}/{len(bundle.val_engine_ids)}/{len(bundle.test)} "
        f"windows(train/val/test)={len(bundle.train)}/{len(bundle.val)}/{len(bundle.test)}"
    )


def main() -> None:
    args = parse_args()
    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        audit(dataset, args)


if __name__ == "__main__":
    main()
