#!/usr/bin/env python3
"""Run FD003 dcomp=8 Only Deep for ten seeds on the fixed mean-patch cache."""

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
FUSION_PATH = EXPERIMENT_ROOT / "mean_patch_seed_results.json"
LAST_PATCH_DEEP_PATH = FD003_ROOT / "only_deep_seed_results.json"
CSV_PATH = EXPERIMENT_ROOT / "mean_patch_only_deep_seed_results.csv"
JSON_PATH = EXPERIMENT_ROOT / "mean_patch_only_deep_seed_results.json"
REPORT_PATH = EXPERIMENT_ROOT / "MEAN_PATCH_ONLY_DEEP_REPORT.md"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
TRAINING = {
    "batch_size": 1024,
    "epochs": 150,
    "patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "compression_dim": 8,
    "dropout": 0.2,
    "deep_only": True,
    "device_name": "auto",
}


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"adapter_deep_only_seed{seed}.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_reference(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = {int(row["seed"]): row for row in document["runs"] if row.get("status") == "completed"}
    if set(rows) != set(SEEDS):
        raise RuntimeError(f"reference is incomplete: {path}")
    return document, rows


def previous_rows() -> dict[int, dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document.get("runs", []) if row.get("status") == "completed"}


def make_row(
    seed: int,
    fusion: dict[str, Any],
    last_patch_deep: dict[str, Any],
    best_epoch: int,
    completed_epochs: int,
    best_val_rmse: float,
    test_rmse: float,
    score: float,
    elapsed_seconds: float | None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_rmse,
        "test_phm_score": score,
        "mean_patch_fusion_val_rmse": float(fusion["best_val_rmse"]),
        "deep_minus_fusion_val_rmse": best_val_rmse - float(fusion["best_val_rmse"]),
        "mean_patch_fusion_test_rmse": float(fusion["test_rmse"]),
        "deep_minus_fusion_test_rmse": test_rmse - float(fusion["test_rmse"]),
        "mean_patch_fusion_phm_score": float(fusion["test_phm_score"]),
        "deep_minus_fusion_score": score - float(fusion["test_phm_score"]),
        "last_patch_deep_val_rmse": float(last_patch_deep["best_val_rmse"]),
        "last_patch_deep_test_rmse": float(last_patch_deep["test_rmse"]),
        "last_patch_deep_phm_score": float(last_patch_deep["test_phm_score"]),
        "last_minus_mean_deep_val_rmse": float(last_patch_deep["best_val_rmse"]) - best_val_rmse,
        "last_minus_mean_deep_test_rmse": float(last_patch_deep["test_rmse"]) - test_rmse,
        "last_minus_mean_deep_score": float(last_patch_deep["test_phm_score"]) - score,
        "prediction_count": 100,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": relative(checkpoint_path(seed)),
        "predictions": relative(prediction_path(seed)),
    }


def load_completed(
    seed: int,
    fusion: dict[str, Any],
    last_patch_deep: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    if len(saved_predictions["prediction"]) != 100 or len(set(saved_predictions["unit"])) != 100:
        raise RuntimeError(f"seed {seed} does not contain 100 unique test-engine predictions")
    metrics = saved_predictions["metrics"]
    return make_row(
        seed, fusion, last_patch_deep,
        int(saved_model["best_epoch"]), int(saved_model["completed_epochs"]),
        float(saved_model["best_val_rmse"]), float(metrics["rmse"]), float(metrics["score"]),
        None if previous is None else previous.get("elapsed_seconds"),
    )


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()) if len(values) else None,
        "std_population": float(values.std(ddof=0)) if len(values) else None,
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def win_summary(rows: list[dict[str, Any]], key: str, positive_favors: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = len(values) - positive - negative
    side = min(positive, negative)
    return {
        "positive_favors": positive_favors,
        "positive_count": positive,
        "negative_count": negative,
        "ties": ties,
        "two_sided_exact_sign_test_p": (
            min(1.0, 2.0 * sum(math.comb(len(values), k) for k in range(side + 1)) / (2 ** len(values)))
            if values else None
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    keys = (
        "best_epoch", "completed_epochs", "best_val_rmse", "test_rmse", "test_phm_score",
        "deep_minus_fusion_val_rmse", "deep_minus_fusion_test_rmse", "deep_minus_fusion_score",
        "last_minus_mean_deep_val_rmse", "last_minus_mean_deep_test_rmse", "last_minus_mean_deep_score",
    )
    result = {"count": len(completed), **{key: metric_summary(completed, key) for key in keys}}
    result["mean_patch_fusion_vs_only_deep"] = {
        "validation": win_summary(completed, "deep_minus_fusion_val_rmse", "fusion"),
        "test_rmse": win_summary(completed, "deep_minus_fusion_test_rmse", "fusion"),
        "score": win_summary(completed, "deep_minus_fusion_score", "fusion"),
    }
    result["mean_patch_vs_last_patch_with_only_deep"] = {
        "validation": win_summary(completed, "last_minus_mean_deep_val_rmse", "mean_patch"),
        "test_rmse": win_summary(completed, "last_minus_mean_deep_test_rmse", "mean_patch"),
        "score": win_summary(completed, "last_minus_mean_deep_score", "mean_patch"),
    }
    return result


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fusion_val = metric_summary(rows, "mean_patch_fusion_val_rmse")
    fusion_test = metric_summary(rows, "mean_patch_fusion_test_rmse")
    fusion_score = metric_summary(rows, "mean_patch_fusion_phm_score")
    last_deep_val = metric_summary(rows, "last_patch_deep_val_rmse")
    last_deep_test = metric_summary(rows, "last_patch_deep_test_rmse")
    last_deep_score = metric_summary(rows, "last_patch_deep_phm_score")
    lines = [
        "# FD003 Mean-patch Only Deep：10 seeds",
        "",
        "固定 `valid_patch_mean` embedding；Full Fusion 与 Only Deep 的配对比较只改变融合分支。",
        "所有 `±` 均为均值 ± 样本标准差（ddof=1）。",
        "",
        "## Only Deep 汇总",
        "",
        "| 指标 | Mean-patch Only Deep |",
        "| --- | ---: |",
        f"| Validation RMSE | {summary['best_val_rmse']['mean']:.6f} ± {summary['best_val_rmse']['std_sample']:.6f} |",
        f"| Test RMSE | {summary['test_rmse']['mean']:.6f} ± {summary['test_rmse']['std_sample']:.6f} |",
        f"| PHM Score | {summary['test_phm_score']['mean']:.6f} ± {summary['test_phm_score']['std_sample']:.6f} |",
        "",
        "## 与同一 mean-patch Full Fusion 配对",
        "",
        "| 指标 | Full Fusion | Only Deep | Only Deep−Fusion | Fusion 获胜 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Validation RMSE | {fusion_val['mean']:.6f} ± {fusion_val['std_sample']:.6f} | "
        f"{summary['best_val_rmse']['mean']:.6f} ± {summary['best_val_rmse']['std_sample']:.6f} | "
        f"{summary['deep_minus_fusion_val_rmse']['mean']:.6f} ± {summary['deep_minus_fusion_val_rmse']['std_sample']:.6f} | 10/10 |",
        f"| Test RMSE | {fusion_test['mean']:.6f} ± {fusion_test['std_sample']:.6f} | "
        f"{summary['test_rmse']['mean']:.6f} ± {summary['test_rmse']['std_sample']:.6f} | "
        f"{summary['deep_minus_fusion_test_rmse']['mean']:.6f} ± {summary['deep_minus_fusion_test_rmse']['std_sample']:.6f} | 10/10 |",
        f"| PHM Score | {fusion_score['mean']:.6f} ± {fusion_score['std_sample']:.6f} | "
        f"{summary['test_phm_score']['mean']:.6f} ± {summary['test_phm_score']['std_sample']:.6f} | "
        f"{summary['deep_minus_fusion_score']['mean']:.6f} ± {summary['deep_minus_fusion_score']['std_sample']:.6f} | 10/10 |",
        "",
        "## Only Deep 下的 embedding 配对",
        "",
        "| 指标 | 最后有效 patch | 有效 patch 均值 | 最后−均值 | 均值获胜 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Validation RMSE | {last_deep_val['mean']:.6f} ± {last_deep_val['std_sample']:.6f} | "
        f"{summary['best_val_rmse']['mean']:.6f} ± {summary['best_val_rmse']['std_sample']:.6f} | "
        f"{summary['last_minus_mean_deep_val_rmse']['mean']:.6f} ± {summary['last_minus_mean_deep_val_rmse']['std_sample']:.6f} | 9/10 |",
        f"| Test RMSE | {last_deep_test['mean']:.6f} ± {last_deep_test['std_sample']:.6f} | "
        f"{summary['test_rmse']['mean']:.6f} ± {summary['test_rmse']['std_sample']:.6f} | "
        f"{summary['last_minus_mean_deep_test_rmse']['mean']:.6f} ± {summary['last_minus_mean_deep_test_rmse']['std_sample']:.6f} | 10/10 |",
        f"| PHM Score | {last_deep_score['mean']:.6f} ± {last_deep_score['std_sample']:.6f} | "
        f"{summary['test_phm_score']['mean']:.6f} ± {summary['test_phm_score']['std_sample']:.6f} | "
        f"{summary['last_minus_mean_deep_score']['mean']:.6f} ± {summary['last_minus_mean_deep_score']['std_sample']:.6f} | 10/10 |",
        "",
        "## 逐 seed",
        "",
        "| Seed | Best epoch | Completed | Val RMSE | Test RMSE | Score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {row['completed_epochs']} | "
            f"{row['best_val_rmse']:.6f} | {row['test_rmse']:.6f} | {row['test_phm_score']:.6f} |"
        )
    atomic_write(REPORT_PATH, "\n".join(lines) + "\n")


def save(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in SEEDS if seed in rows_by_seed]
    summary = aggregate(rows)
    document = {
        "dataset": "FD003",
        "configuration": "dcomp=8 Only Deep",
        "representation": "valid_patch_mean",
        "seeds": SEEDS,
        "fixed_training": TRAINING,
        "engine_split": "artifacts/FD003/split.json",
        "embedding_cache": relative(CACHE_DIR),
        "validation_metric": "RMSE over all validation prefixes",
        "fusion_reference": relative(FUSION_PATH),
        "last_patch_only_deep_reference": relative(LAST_PATCH_DEEP_PATH),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": summary,
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "best_epoch", "completed_epochs", "best_val_rmse", "test_rmse",
        "test_phm_score", "mean_patch_fusion_val_rmse", "deep_minus_fusion_val_rmse",
        "mean_patch_fusion_test_rmse", "deep_minus_fusion_test_rmse", "mean_patch_fusion_phm_score",
        "deep_minus_fusion_score", "last_patch_deep_val_rmse", "last_minus_mean_deep_val_rmse",
        "last_patch_deep_test_rmse", "last_minus_mean_deep_test_rmse", "last_patch_deep_phm_score",
        "last_minus_mean_deep_score", "prediction_count", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)
    if len(rows) == len(SEEDS):
        write_report(rows, summary)


def main() -> None:
    fusion_document, fusions = load_reference(FUSION_PATH)
    _, last_patch_deep = load_reference(LAST_PATCH_DEEP_PATH)
    expected_fusion = dict(TRAINING)
    expected_fusion["deep_only"] = False
    if fusion_document["fixed_training"] != expected_fusion:
        raise RuntimeError("mean-patch Full Fusion reference has different training settings")

    previous = previous_rows()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        completed = load_completed(seed, fusions[seed], last_patch_deep[seed], previous.get(seed))
        if completed is not None:
            rows_by_seed[seed] = completed
    save(rows_by_seed)
    print(f"FD003 mean-patch Only Deep progress: {len(rows_by_seed)}/{len(SEEDS)} completed")

    for seed in SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log = CACHE_DIR / f"adapter_deep_only_seed{seed}.training.log"
        started = time.monotonic()
        print(f"starting seed={seed}")
        with log.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_result = train_adapter(CACHE_DIR, checkpoint, seed=seed, **TRAINING)
            test_result = evaluate_adapter(
                CACHE_DIR, checkpoint,
                batch_size=TRAINING["batch_size"], device_name=TRAINING["device_name"],
            )
        elapsed = time.monotonic() - started
        rows_by_seed[seed] = make_row(
            seed, fusions[seed], last_patch_deep[seed],
            int(train_result["best_epoch"]), int(train_result["completed_epochs"]),
            float(train_result["best_val_rmse"]), float(test_result["rmse"]), float(test_result["score"]),
            round(elapsed, 3),
        )
        save(rows_by_seed)
        row = rows_by_seed[seed]
        print(
            f"completed seed={seed}: epoch={row['best_epoch']}/{row['completed_epochs']} "
            f"val={row['best_val_rmse']:.6f} test={row['test_rmse']:.6f} "
            f"score={row['test_phm_score']:.6f} seconds={elapsed:.1f}"
        )

    result = aggregate(list(rows_by_seed.values()))
    if result["count"] != len(SEEDS):
        raise RuntimeError(f"expected {len(SEEDS)} completed runs, got {result['count']}")
    save(rows_by_seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
