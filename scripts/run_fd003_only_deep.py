#!/usr/bin/env python3
"""Run the FD003 dcomp=8 Only Deep ablation for ten fixed seeds."""

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
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD003"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
CSV_PATH = CACHE_DIR / "only_deep_seed_results.csv"
JSON_PATH = CACHE_DIR / "only_deep_seed_results.json"
FUSION_PATH = CACHE_DIR / "seed_results.json"

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


def fusion_rows() -> dict[int, dict[str, Any]]:
    document = json.loads(FUSION_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document["runs"]}


def previous_rows() -> dict[int, dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document.get("runs", []) if row.get("status") == "completed"}


def make_row(
    seed: int,
    fusion: dict[str, Any],
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
        "fusion_test_rmse": float(fusion["test_rmse"]),
        "deep_minus_fusion_test_rmse": test_rmse - float(fusion["test_rmse"]),
        "fusion_phm_score": float(fusion["test_phm_score"]),
        "deep_minus_fusion_score": score - float(fusion["test_phm_score"]),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(checkpoint_path(seed).relative_to(PROJECT_ROOT)),
        "predictions": str(prediction_path(seed).relative_to(PROJECT_ROOT)),
    }


def load_completed(seed: int, fusion: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    metrics = saved_predictions["metrics"]
    return make_row(
        seed,
        fusion,
        int(saved_model["best_epoch"]),
        int(saved_model["completed_epochs"]),
        float(saved_model["best_val_rmse"]),
        float(metrics["rmse"]),
        float(metrics["score"]),
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


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    result: dict[str, Any] = {"count": len(completed)}
    for key in (
        "best_epoch", "completed_epochs", "best_val_rmse", "test_rmse", "test_phm_score",
        "deep_minus_fusion_test_rmse", "deep_minus_fusion_score",
    ):
        result[key] = metric_summary(completed, key)
    paired_count = len(completed)
    fusion_wins = sum(row["deep_minus_fusion_test_rmse"] > 0 for row in completed)
    losses = paired_count - fusion_wins
    side = min(fusion_wins, losses)
    result["paired_comparison"] = {
        "fusion_rmse_wins": fusion_wins,
        "only_deep_rmse_wins": losses,
        "two_sided_exact_sign_test_p": (
            min(1.0, 2.0 * sum(math.comb(paired_count, k) for k in range(side + 1)) / (2**paired_count))
            if paired_count else None
        ),
    }
    return result


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
    document = {
        "dataset": "FD003",
        "configuration": "dcomp=8 Only Deep",
        "seeds": SEEDS,
        "fixed_training": TRAINING,
        "engine_split": "artifacts/FD003/split.json",
        "embedding_cache": "same frozen Chronos-2 cache as Full Fusion",
        "validation_metric": "RMSE over all validation prefixes",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": aggregate(rows),
        "paper": {"only_deep_fd003_rmse_mean": 10.16, "only_deep_fd003_rmse_std": 0.19},
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "best_epoch", "completed_epochs", "best_val_rmse",
        "test_rmse", "test_phm_score", "fusion_test_rmse", "deep_minus_fusion_test_rmse",
        "fusion_phm_score", "deep_minus_fusion_score", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)


def main() -> None:
    fusions = fusion_rows()
    previous = previous_rows()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        completed = load_completed(seed, fusions[seed], previous.get(seed))
        if completed is not None:
            rows_by_seed[seed] = completed
    save(rows_by_seed)
    print(f"FD003 Only Deep progress: {len(rows_by_seed)}/{len(SEEDS)} completed")

    for seed in SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log_path = CACHE_DIR / f"adapter_deep_only_seed{seed}.training.log"
        started = time.monotonic()
        print(f"starting seed={seed}")
        try:
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                train_result = train_adapter(CACHE_DIR, checkpoint, seed=seed, **TRAINING)
                test_result = evaluate_adapter(
                    CACHE_DIR,
                    checkpoint,
                    batch_size=TRAINING["batch_size"],
                    device_name=TRAINING["device_name"],
                )
            elapsed = time.monotonic() - started
            rows_by_seed[seed] = make_row(
                seed,
                fusions[seed],
                int(train_result["best_epoch"]),
                int(train_result["completed_epochs"]),
                float(train_result["best_val_rmse"]),
                float(test_result["rmse"]),
                float(test_result["score"]),
                round(elapsed, 3),
            )
            save(rows_by_seed)
            row = rows_by_seed[seed]
            print(
                f"completed seed={seed}: best_epoch={row['best_epoch']} completed={row['completed_epochs']} "
                f"val={row['best_val_rmse']:.4f} test={row['test_rmse']:.4f} "
                f"score={row['test_phm_score']:.2f} "
                f"deep-fusion={row['deep_minus_fusion_test_rmse']:+.4f} seconds={elapsed:.1f}"
            )
        except Exception as error:
            rows_by_seed[seed] = {
                "seed": seed,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "predictions": str(prediction_path(seed).relative_to(PROJECT_ROOT)),
            }
            save(rows_by_seed)
            raise

    save(rows_by_seed)
    result = aggregate(list(rows_by_seed.values()))
    if result["count"] != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} completed runs, got {result['count']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
