#!/usr/bin/env python3
"""Extend three predeclared FD003 seeds from max 150 to max 300 epochs."""

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
SEEDS = [0, 1, 2]
CSV_PATH = CACHE_DIR / "max300_results.csv"
JSON_PATH = CACHE_DIR / "max300_results.json"
BASELINE_PATH = CACHE_DIR / "seed_results.json"

TRAINING = {
    "batch_size": 1024,
    "epochs": 300,
    "patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "compression_dim": 8,
    "dropout": 0.2,
    "deep_only": False,
    "device_name": "auto",
}


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"adapter_wide_deep_seed{seed}_max300.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def baseline_rows() -> dict[int, dict[str, Any]]:
    document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document["runs"]}


def previous_rows() -> dict[int, dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in document.get("runs", []) if row.get("status") == "completed"}


def load_completed(seed: int, baseline: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    metrics = saved_predictions["metrics"]
    return make_row(
        seed,
        baseline,
        int(saved_model["best_epoch"]),
        int(saved_model["completed_epochs"]),
        float(saved_model["best_val_rmse"]),
        float(metrics["rmse"]),
        float(metrics["score"]),
        None if previous is None else previous.get("elapsed_seconds"),
    )


def make_row(
    seed: int,
    baseline: dict[str, Any],
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
        "max150_best_epoch": int(baseline["best_epoch"]),
        "max300_best_epoch": best_epoch,
        "max300_completed_epochs": completed_epochs,
        "max150_best_val_rmse": float(baseline["best_val_rmse"]),
        "max300_best_val_rmse": best_val_rmse,
        "val_rmse_change": best_val_rmse - float(baseline["best_val_rmse"]),
        "max150_test_rmse": float(baseline["test_rmse"]),
        "max300_test_rmse": test_rmse,
        "test_rmse_change": test_rmse - float(baseline["test_rmse"]),
        "max150_score": float(baseline["test_phm_score"]),
        "max300_score": score,
        "score_change": score - float(baseline["test_phm_score"]),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(checkpoint_path(seed).relative_to(PROJECT_ROOT)),
        "predictions": str(prediction_path(seed).relative_to(PROJECT_ROOT)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    result: dict[str, Any] = {"count": len(completed)}
    for key in ("max300_best_epoch", "max300_completed_epochs", "val_rmse_change", "test_rmse_change", "score_change"):
        values = np.asarray([row[key] for row in completed], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    result["best_epoch_after_150_count"] = sum(row["max300_best_epoch"] > 150 for row in completed)
    result["test_rmse_improved_count"] = sum(row["test_rmse_change"] < 0 for row in completed)
    return result


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
    document = {
        "dataset": "FD003",
        "seeds": SEEDS,
        "experiment": "same dcomp=8 Full Fusion configuration with max_epochs changed from 150 to 300",
        "fixed_training": TRAINING,
        "resume_note": "retrained deterministically from epoch 1 because max150 checkpoints do not contain last-state optimizer data",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "max150_best_epoch", "max300_best_epoch", "max300_completed_epochs",
        "max150_best_val_rmse", "max300_best_val_rmse", "val_rmse_change",
        "max150_test_rmse", "max300_test_rmse", "test_rmse_change",
        "max150_score", "max300_score", "score_change", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)


def main() -> None:
    baselines = baseline_rows()
    previous = previous_rows()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        completed = load_completed(seed, baselines[seed], previous.get(seed))
        if completed is not None:
            rows_by_seed[seed] = completed
    save(rows_by_seed)
    print(f"max300 progress: {len(rows_by_seed)}/{len(SEEDS)} completed")

    for seed in SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log_path = CACHE_DIR / f"adapter_wide_deep_seed{seed}_max300.training.log"
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
                baselines[seed],
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
                f"completed seed={seed}: best_epoch={row['max300_best_epoch']} "
                f"completed_epochs={row['max300_completed_epochs']} "
                f"val_change={row['val_rmse_change']:+.4f} "
                f"test_change={row['test_rmse_change']:+.4f} score_change={row['score_change']:+.2f} "
                f"seconds={elapsed:.1f}"
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
