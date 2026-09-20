#!/usr/bin/env python3
"""Run FD003 valid-patch-mean for ten seeds and pair with the last-patch baseline."""

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

from rul_chronos.training import evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FD003_ROOT = PROJECT_ROOT / "artifacts" / "FD003"
EXPERIMENT_ROOT = FD003_ROOT / "patch_representation_ablation"
CACHE_DIR = EXPERIMENT_ROOT / "valid_patch_mean"
BASELINE_PATH = FD003_ROOT / "seed_results.json"
RESULTS_JSON = EXPERIMENT_ROOT / "mean_patch_seed_results.json"
RESULTS_CSV = EXPERIMENT_ROOT / "mean_patch_seed_results.csv"
PAIRED_JSON = EXPERIMENT_ROOT / "paired_comparison.json"
PAIRED_CSV = EXPERIMENT_ROOT / "paired_comparison.csv"
REPORT_PATH = EXPERIMENT_ROOT / "TEN_SEED_REPORT.md"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
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
}
METRICS = ("best_val_rmse", "test_rmse", "test_phm_score")


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"adapter_wide_deep_seed{seed}.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def metric_summary(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def load_previous() -> dict[int, dict[str, Any]]:
    if not RESULTS_JSON.exists():
        return {}
    document = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document.get("runs", []) if row.get("status") == "completed"}


def load_completed(seed: int, previous: dict[str, Any] | None) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    if len(saved_predictions["prediction"]) != 100 or len(set(saved_predictions["unit"])) != 100:
        raise RuntimeError(f"seed {seed} does not contain 100 unique test-engine predictions")
    metrics = saved_predictions["metrics"]
    return {
        "seed": seed,
        "status": "completed",
        "best_epoch": int(saved_model["best_epoch"]),
        "completed_epochs": int(saved_model["completed_epochs"]),
        "best_val_rmse": float(saved_model["best_val_rmse"]),
        "test_rmse": float(metrics["rmse"]),
        "test_phm_score": float(metrics["score"]),
        "prediction_count": len(saved_predictions["prediction"]),
        "elapsed_seconds": None if previous is None else previous.get("elapsed_seconds"),
        "checkpoint": relative(checkpoint),
        "predictions": relative(predictions),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    return {
        "count": len(completed),
        **{metric: metric_summary([float(row[metric]) for row in completed]) for metric in METRICS},
    }


def save_results(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in SEEDS if seed in rows_by_seed]
    document = {
        "dataset": "FD003",
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion",
        "seeds": SEEDS,
        "fixed_training": TRAINING,
        "validation_metric": "RMSE over all validation prefixes",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    atomic_write(RESULTS_JSON, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "best_epoch", "completed_epochs", "best_val_rmse",
        "test_rmse", "test_phm_score", "prediction_count", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = RESULTS_CSV.with_suffix(RESULTS_CSV.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(RESULTS_CSV)


def load_baseline() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if document["seeds"] != SEEDS:
        raise RuntimeError("last-patch baseline seeds differ from the requested seed list")
    if document["fixed_training"] != TRAINING:
        raise RuntimeError("last-patch baseline training configuration differs")
    rows = {int(row["seed"]): row for row in document["runs"] if row.get("status") == "completed"}
    if set(rows) != set(SEEDS):
        raise RuntimeError("last-patch baseline is not complete for all ten seeds")
    return document, rows


def build_paired(
    mean_rows: dict[int, dict[str, Any]],
    baseline_rows: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired = []
    for seed in SEEDS:
        last = baseline_rows[seed]
        mean = mean_rows[seed]
        row: dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            row[f"last_patch_{metric}"] = float(last[metric])
            row[f"mean_patch_{metric}"] = float(mean[metric])
            row[f"last_minus_mean_{metric}"] = float(last[metric]) - float(mean[metric])
        paired.append(row)

    comparison: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [float(row[f"last_minus_mean_{metric}"]) for row in paired]
        comparison[metric] = {
            "last_valid_patch": metric_summary([float(row[f"last_patch_{metric}"]) for row in paired]),
            "valid_patch_mean": metric_summary([float(row[f"mean_patch_{metric}"]) for row in paired]),
            "paired_last_minus_mean": metric_summary(deltas),
            "mean_patch_wins": sum(delta > 0 for delta in deltas),
            "last_patch_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(math.isclose(delta, 0.0, abs_tol=1e-12) for delta in deltas),
        }
    return paired, comparison


def save_paired(paired: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    document = {
        "dataset": "FD003",
        "seeds": SEEDS,
        "design": "paired by seed; positive last_minus_mean values favor valid_patch_mean",
        "last_patch_source": relative(BASELINE_PATH),
        "mean_patch_source": relative(RESULTS_JSON),
        "standard_deviation_note": "std_sample uses ddof=1; std_population uses ddof=0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paired_runs": paired,
        "comparison": comparison,
    }
    atomic_write(PAIRED_JSON, json.dumps(document, indent=2) + "\n")
    fields = ["seed"]
    for metric in METRICS:
        fields.extend((f"last_patch_{metric}", f"mean_patch_{metric}", f"last_minus_mean_{metric}"))
    temporary = PAIRED_CSV.with_suffix(PAIRED_CSV.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in paired)
    temporary.replace(PAIRED_CSV)

    labels = {
        "best_val_rmse": "Validation RMSE",
        "test_rmse": "Test RMSE",
        "test_phm_score": "PHM Score",
    }
    lines = [
        "# FD003 有效 patch 均值：10 seeds 配对实验",
        "",
        "固定配置：Full Fusion、`dcomp=8`、AdamW、学习率 `1e-4`、权重衰减 `1e-3`、",
        "batch size 1024、dropout 0.2、最多 150 epochs、patience 15。",
        "所有结果使用同一份 80/20 发动机划分；标准差为样本标准差（ddof=1）。",
        "",
        "## 汇总",
        "",
        "| 指标 | 最后有效 patch | 有效 patch 均值 | 配对差值（最后−均值） | 均值表示获胜 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        item = comparison[metric]
        last = item["last_valid_patch"]
        mean = item["valid_patch_mean"]
        delta = item["paired_last_minus_mean"]
        lines.append(
            f"| {labels[metric]} | {last['mean']:.6f} ± {last['std_sample']:.6f} | "
            f"{mean['mean']:.6f} ± {mean['std_sample']:.6f} | "
            f"{delta['mean']:.6f} ± {delta['std_sample']:.6f} | "
            f"{item['mean_patch_wins']}/10 |"
        )
    lines.extend([
        "",
        "正的配对差值表示有效 patch 均值更低、表现更好。验证集胜负次数用于判断提升是否只出现在测试集。",
        "",
        "## 逐 seed",
        "",
        "| Seed | Val last | Val mean | Test last | Test mean | Score last | Score mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in paired:
        lines.append(
            f"| {row['seed']} | {row['last_patch_best_val_rmse']:.6f} | "
            f"{row['mean_patch_best_val_rmse']:.6f} | {row['last_patch_test_rmse']:.6f} | "
            f"{row['mean_patch_test_rmse']:.6f} | {row['last_patch_test_phm_score']:.6f} | "
            f"{row['mean_patch_test_phm_score']:.6f} |"
        )
    atomic_write(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    _, baseline_rows = load_baseline()
    previous = load_previous()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        completed = load_completed(seed, previous.get(seed))
        if completed is not None:
            rows_by_seed[seed] = completed
    save_results(rows_by_seed)
    print(f"FD003 valid-patch-mean progress: {len(rows_by_seed)}/{len(SEEDS)} completed")

    for seed in SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log = CACHE_DIR / f"adapter_wide_deep_seed{seed}.training.log"
        started = time.monotonic()
        print(f"starting seed={seed}")
        with log.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_result = train_adapter(CACHE_DIR, checkpoint, seed=seed, **TRAINING)
            test_result = evaluate_adapter(
                CACHE_DIR,
                checkpoint,
                batch_size=TRAINING["batch_size"],
                device_name=TRAINING["device_name"],
            )
        elapsed = time.monotonic() - started
        rows_by_seed[seed] = {
            "seed": seed,
            "status": "completed",
            "best_epoch": int(train_result["best_epoch"]),
            "completed_epochs": int(train_result["completed_epochs"]),
            "best_val_rmse": float(train_result["best_val_rmse"]),
            "test_rmse": float(test_result["rmse"]),
            "test_phm_score": float(test_result["score"]),
            "prediction_count": 100,
            "elapsed_seconds": round(elapsed, 3),
            "checkpoint": relative(checkpoint),
            "predictions": relative(prediction_path(seed)),
        }
        save_results(rows_by_seed)
        row = rows_by_seed[seed]
        print(
            f"completed seed={seed}: epoch={row['best_epoch']}/{row['completed_epochs']} "
            f"val={row['best_val_rmse']:.6f} test={row['test_rmse']:.6f} "
            f"score={row['test_phm_score']:.6f} seconds={elapsed:.1f}"
        )

    if set(rows_by_seed) != set(SEEDS):
        raise RuntimeError("mean-patch experiment is incomplete")
    paired, comparison = build_paired(rows_by_seed, baseline_rows)
    save_paired(paired, comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
