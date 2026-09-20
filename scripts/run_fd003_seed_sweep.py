#!/usr/bin/env python3
"""Run the fixed FD003 dcomp=8 Full Fusion configuration for ten seeds."""

from __future__ import annotations

import csv
import json
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
NEW_SEEDS = list(range(9))
CSV_PATH = CACHE_DIR / "seed_results.csv"
JSON_PATH = CACHE_DIR / "seed_results.json"

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


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"adapter_wide_deep_seed{seed}.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def existing_summary() -> dict[str, Any]:
    path = CACHE_DIR / "run_summary_seed42.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_previous_rows() -> dict[int, dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document.get("runs", []) if row.get("status") == "completed"}


def load_completed(seed: int, previous: dict[str, Any] | None, seed42_summary: dict[str, Any]) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    metrics = saved_predictions["metrics"]
    best_epoch = saved_model.get("best_epoch")
    completed_epochs = saved_model.get("completed_epochs")
    if seed == 42 and best_epoch is None:
        adapter = seed42_summary.get("adapter", {})
        best_epoch = adapter.get("best_epoch")
        completed_epochs = adapter.get("completed_epochs")
    if best_epoch is None or completed_epochs is None:
        raise RuntimeError(f"Missing epoch metadata for completed seed {seed}")
    return {
        "seed": seed,
        "status": "completed",
        "best_epoch": int(best_epoch),
        "completed_epochs": int(completed_epochs),
        "best_val_rmse": float(saved_model["best_val_rmse"]),
        "test_rmse": float(metrics["rmse"]),
        "test_phm_score": float(metrics["score"]),
        "elapsed_seconds": None if previous is None else previous.get("elapsed_seconds"),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "predictions": str(predictions.relative_to(PROJECT_ROOT)),
    }


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()) if len(values) else None,
        "std_population": float(values.std(ddof=0)) if len(values) else None,
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    result: dict[str, Any] = {"count": len(completed)}
    for key in ("best_epoch", "completed_epochs", "best_val_rmse", "test_rmse", "test_phm_score"):
        result[key] = metric_summary(completed, key)
    if len(completed) == len(SEEDS):
        by_seed = {row["seed"]: row for row in completed}
        seed42 = by_seed[42]
        result["diagnostics"] = {
            "seed42_test_rmse_rank_ascending": 1 + sum(row["test_rmse"] < seed42["test_rmse"] for row in completed),
            "seed42_score_rank_ascending": 1 + sum(row["test_phm_score"] < seed42["test_phm_score"] for row in completed),
            "best_epoch_at_least_140_count": sum(row["best_epoch"] >= 140 for row in completed),
            "best_epoch_at_least_145_count": sum(row["best_epoch"] >= 145 for row in completed),
            "ran_full_150_epochs_count": sum(row["completed_epochs"] == 150 for row in completed),
            "early_stopped_count": sum(row["completed_epochs"] < 150 for row in completed),
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
        "configuration": "dcomp=8 Full Fusion",
        "seeds": SEEDS,
        "new_seeds": NEW_SEEDS,
        "fixed_training": TRAINING,
        "validation_metric": "RMSE over all validation prefixes",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": summarize(rows),
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "best_epoch", "completed_epochs", "best_val_rmse",
        "test_rmse", "test_phm_score", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)


def main() -> None:
    previous = load_previous_rows()
    seed42_summary = existing_summary()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        row = load_completed(seed, previous.get(seed), seed42_summary)
        if row is not None:
            rows_by_seed[seed] = row
    save(rows_by_seed)
    print(f"FD003 progress: {len(rows_by_seed)}/{len(SEEDS)} runs already completed")

    for seed in NEW_SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log_path = CACHE_DIR / f"adapter_wide_deep_seed{seed}.training.log"
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
            rows_by_seed[seed] = {
                "seed": seed,
                "status": "completed",
                "best_epoch": int(train_result["best_epoch"]),
                "completed_epochs": int(train_result["completed_epochs"]),
                "best_val_rmse": float(train_result["best_val_rmse"]),
                "test_rmse": float(test_result["rmse"]),
                "test_phm_score": float(test_result["score"]),
                "elapsed_seconds": round(elapsed, 3),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "predictions": str(prediction_path(seed).relative_to(PROJECT_ROOT)),
            }
            save(rows_by_seed)
            row = rows_by_seed[seed]
            print(
                f"completed seed={seed}: best_epoch={row['best_epoch']} completed_epochs={row['completed_epochs']} "
                f"val={row['best_val_rmse']:.4f} test={row['test_rmse']:.4f} "
                f"score={row['test_phm_score']:.2f} seconds={elapsed:.1f}"
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
    aggregate = summarize(list(rows_by_seed.values()))
    if aggregate["count"] != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} completed runs, got {aggregate['count']}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
