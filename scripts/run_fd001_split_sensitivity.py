#!/usr/bin/env python3
"""Run the pre-registered nine-run FD001 split-sensitivity experiment."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rul_chronos.metrics import phm_score, rmse
from rul_chronos.training import evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "artifacts" / "FD001" / "split_sensitivity"
CURRENT_CACHE = PROJECT_ROOT / "artifacts" / "FD001" / "representation_ablation" / "valid_patch_mean"
SPLIT_SEEDS = (42, 2026, 2027)
TRAINING_SEEDS = (0, 1, 42)
CSV_PATH = ROOT / "split_sensitivity_runs.csv"
JSON_PATH = ROOT / "split_sensitivity_summary.json"
REPORT_PATH = ROOT / "SPLIT_SENSITIVITY_REPORT.md"
TRAINING = {
    "batch_size": 1024,
    "epochs": 150,
    "patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "compression_dim": 8,
    "dropout": 0.2,
    "deep_only": False,
    "device_name": "auto",
    "min_prefix_length": 1,
}
METRICS = ("best_val_rmse", "test_rmse", "test_phm_score")


def cache_dir(split_seed: int) -> Path:
    return CURRENT_CACHE if split_seed == 42 else ROOT / f"split_seed{split_seed}"


def checkpoint_path(split_seed: int, training_seed: int) -> Path:
    if split_seed == 42:
        return CURRENT_CACHE / f"adapter_wide_deep_seed{training_seed}.pt"
    return cache_dir(split_seed) / f"adapter_wide_deep_trainseed{training_seed}.pt"


def prediction_path(split_seed: int, training_seed: int) -> Path:
    return checkpoint_path(split_seed, training_seed).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def previous_rows() -> dict[tuple[int, int], dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {
        (int(row["split_seed"]), int(row["training_seed"])): row
        for row in document.get("runs", [])
        if row.get("status") == "completed"
    }


def load_completed(
    split_seed: int, training_seed: int, previous: dict[str, Any] | None
) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(split_seed, training_seed)
    predictions = prediction_path(split_seed, training_seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_kwargs = {
        "num_sensors": 21,
        "embedding_dim": 768,
        "compression_dim": 8,
        "num_regimes": 1,
        "dropout": 0.2,
        "deep_only": False,
    }
    if int(saved["seed"]) != training_seed or saved["model_kwargs"] != expected_kwargs:
        raise RuntimeError(f"checkpoint configuration differs for {(split_seed, training_seed)}")
    if int(saved.get("min_prefix_length", 1)) != 1:
        raise RuntimeError("unexpected prefix restriction")
    data = json.loads(predictions.read_text(encoding="utf-8"))
    units = np.asarray(data["unit"], dtype=np.int64)
    target = np.asarray(data["target"], dtype=np.float64)
    prediction = np.asarray(data["prediction"], dtype=np.float64)
    if len(prediction) != 100 or len(np.unique(units)) != 100 or not np.isfinite(prediction).all():
        raise RuntimeError(f"invalid official-test predictions for {(split_seed, training_seed)}")
    test_rmse = rmse(target, prediction)
    score = phm_score(target, prediction)
    if not np.isclose(test_rmse, data["metrics"]["rmse"], rtol=0.0, atol=1e-10):
        raise RuntimeError("stored test RMSE differs")
    if not np.isclose(score, data["metrics"]["score"], rtol=0.0, atol=1e-10):
        raise RuntimeError("stored PHM Score differs")
    directory = cache_dir(split_seed)
    train_size = json.loads((directory / "train_metadata.json").read_text())["size"]
    validation_size = json.loads((directory / "val_metadata.json").read_text())["size"]
    return {
        "split_seed": split_seed,
        "training_seed": training_seed,
        "status": "completed",
        "reused": split_seed == 42,
        "best_epoch": int(saved["best_epoch"]),
        "completed_epochs": int(saved["completed_epochs"]),
        "best_val_rmse": float(saved["best_val_rmse"]),
        "test_rmse": float(test_rmse),
        "test_phm_score": float(score),
        "train_samples": int(saved.get("train_samples", train_size)),
        "validation_samples": int(saved.get("validation_samples", validation_size)),
        "prediction_count": int(len(prediction)),
        "unique_test_engines": int(len(np.unique(units))),
        "elapsed_seconds": None if previous is None else previous.get("elapsed_seconds"),
        "checkpoint": relative(checkpoint),
        "predictions": relative(predictions),
    }


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for split_seed in SPLIT_SEEDS:
        selected = [row for row in rows if row["split_seed"] == split_seed and row["status"] == "completed"]
        if selected:
            groups.append({
                "split_seed": split_seed,
                "count": len(selected),
                **{metric: summary([float(row[metric]) for row in selected]) for metric in METRICS},
            })
    return groups


def sensitivity(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if len(groups) != 3 or any(group["count"] != len(TRAINING_SEEDS) for group in groups):
        return {}
    lookup = {group["split_seed"]: group for group in groups}
    result = {}
    for metric in METRICS:
        means = np.asarray([group[metric]["mean"] for group in groups], dtype=np.float64)
        within_sds = np.asarray([group[metric]["std_sample"] for group in groups], dtype=np.float64)
        pooled_within_sd = float(np.sqrt(np.mean(np.square(within_sds))))
        comparisons = []
        for new_seed in (2026, 2027):
            shift = float(lookup[new_seed][metric]["mean"] - lookup[42][metric]["mean"])
            pair_scale = math.sqrt(
                (lookup[new_seed][metric]["std_sample"] ** 2 + lookup[42][metric]["std_sample"] ** 2) / 2
            )
            comparisons.append({
                "new_split_seed": new_seed,
                "new_minus_current_mean": shift,
                "absolute_shift": abs(shift),
                "pooled_pair_within_split_sd": pair_scale,
                "absolute_shift_over_pooled_within_sd": abs(shift) / pair_scale if pair_scale else math.inf,
            })
        result[metric] = {
            "split_mean_range": float(means.max() - means.min()),
            "split_means_std_sample": float(means.std(ddof=1)),
            "pooled_within_split_training_seed_sd": pooled_within_sd,
            "range_over_pooled_within_sd": float((means.max() - means.min()) / pooled_within_sd),
            "new_splits_vs_current": comparisons,
        }
    return result


def save(rows_by_key: dict[tuple[int, int], dict[str, Any]]) -> None:
    rows = [rows_by_key[key] for key in ((s, t) for s in SPLIT_SEEDS for t in TRAINING_SEEDS) if key in rows_by_key]
    groups = aggregate(rows)
    analysis = sensitivity(groups)
    document = {
        "dataset": "FD001",
        "purpose": "split sensitivity analysis; not split selection",
        "main_split_seed": 42,
        "supplementary_split_seeds": [2026, 2027],
        "training_seeds_per_split": list(TRAINING_SEEDS),
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion; un-clipped predictions",
        "fixed_training": TRAINING,
        "checkpoint_selection": "lowest validation all-prefix RMSE within each run",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "by_split": groups,
        "sensitivity": analysis,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "split_seed", "training_seed", "status", "reused", "best_epoch", "completed_epochs",
        "best_val_rmse", "test_rmse", "test_phm_score", "train_samples", "validation_samples",
        "prediction_count", "unique_test_engines", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)
    if len(rows) == 9:
        lines = [
            "# FD001 划分敏感性实验",
            "",
            "这是预注册的划分敏感性分析，不用于寻找最佳划分。当前split seed42保留为主结果；",
            "2026和2027只作补充。每个划分固定训练seeds `0, 1, 42`，均值后的 `±` 为样本标准差。",
            "",
            "| Split seed | Val RMSE | Test RMSE | PHM Score |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for group in groups:
            lines.append(
                f"| {group['split_seed']} | {group['best_val_rmse']['mean']:.6f} ± "
                f"{group['best_val_rmse']['std_sample']:.6f} | {group['test_rmse']['mean']:.6f} ± "
                f"{group['test_rmse']['std_sample']:.6f} | {group['test_phm_score']['mean']:.6f} ± "
                f"{group['test_phm_score']['std_sample']:.6f} |"
            )
        lines.extend([
            "",
            "不同划分的验证RMSE评价不同发动机，不用于判断哪个划分更好。",
            "划分间变化用三个划分均值的范围，与划分内训练seed的合并标准差作描述性比较。",
            "",
            "| 指标 | 划分均值范围 | 划分内合并SD | 范围/划分内SD |",
            "| --- | ---: | ---: | ---: |",
        ])
        labels = {"best_val_rmse": "Val RMSE", "test_rmse": "Test RMSE", "test_phm_score": "PHM Score"}
        for metric in METRICS:
            item = analysis[metric]
            lines.append(
                f"| {labels[metric]} | {item['split_mean_range']:.6f} | "
                f"{item['pooled_within_split_training_seed_sd']:.6f} | "
                f"{item['range_over_pooled_within_sd']:.3f} |"
            )
        lines.extend(["", "## 逐次运行", "", "| Split | Train seed | Best epoch | Val RMSE | Test RMSE | Score |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(
                f"| {row['split_seed']} | {row['training_seed']} | {row['best_epoch']} | "
                f"{row['best_val_rmse']:.6f} | {row['test_rmse']:.6f} | {row['test_phm_score']:.6f} |"
            )
        atomic_write(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    for split_seed in (2026, 2027):
        directory = cache_dir(split_seed)
        if not (directory / "manifest.json").exists():
            raise RuntimeError("run scripts/extract_fd001_split_sensitivity.py first")
        split = json.loads((directory / "split.json").read_text())
        if len(split["train_engine_ids"]) != 80 or len(split["val_engine_ids"]) != 20:
            raise RuntimeError("unexpected engine counts")
        if set(split["train_engine_ids"]) & set(split["val_engine_ids"]):
            raise RuntimeError("train/validation engine overlap")

    previous = previous_rows()
    rows_by_key = {}
    for split_seed in SPLIT_SEEDS:
        for training_seed in TRAINING_SEEDS:
            key = (split_seed, training_seed)
            row = load_completed(split_seed, training_seed, previous.get(key))
            if row is not None:
                rows_by_key[key] = row
    save(rows_by_key)
    print(f"split sensitivity progress: {len(rows_by_key)}/9 completed")

    for split_seed in (2026, 2027):
        for training_seed in TRAINING_SEEDS:
            key = (split_seed, training_seed)
            if key in rows_by_key:
                continue
            checkpoint = checkpoint_path(split_seed, training_seed)
            log_path = checkpoint.with_suffix(".training.log")
            started = time.monotonic()
            print(f"starting split_seed={split_seed} training_seed={training_seed}")
            with log_path.open("w", encoding="utf-8") as output, redirect_stdout(output):
                train_adapter(cache_dir(split_seed), checkpoint, seed=training_seed, **TRAINING)
                evaluate_adapter(cache_dir(split_seed), checkpoint, 1024, "auto")
            elapsed = round(time.monotonic() - started, 3)
            row = load_completed(split_seed, training_seed, None)
            assert row is not None
            row["elapsed_seconds"] = elapsed
            rows_by_key[key] = row
            save(rows_by_key)
            print(
                f"completed {len(rows_by_key)}/9: split={split_seed} train_seed={training_seed} "
                f"val={row['best_val_rmse']:.6f} test={row['test_rmse']:.6f} "
                f"score={row['test_phm_score']:.6f}"
            )
    if len(rows_by_key) != 9:
        raise RuntimeError("split sensitivity experiment is incomplete")
    save(rows_by_key)
    print(json.dumps(json.loads(JSON_PATH.read_text())["sensitivity"], indent=2))


if __name__ == "__main__":
    main()
