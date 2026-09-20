#!/usr/bin/env python3
"""Run the FD001 adapter for nine new seeds and persist every result."""

from __future__ import annotations

import csv
import json
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from rul_chronos.training import evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD001"
SEEDS = [*range(9), 42]
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


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def checkpoint_path(seed: int) -> Path:
    return CACHE_DIR / f"adapter_wide_deep_seed{seed}.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def load_completed(seed: int) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    metrics = saved_predictions["metrics"]
    return {
        "seed": seed,
        "status": "completed",
        "best_val_rmse": float(saved_model["best_val_rmse"]),
        "test_rmse": float(metrics["rmse"]),
        "test_phm_score": float(metrics["score"]),
        "elapsed_seconds": None,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "predictions": str(predictions.relative_to(PROJECT_ROOT)),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    summary: dict[str, Any] = {"count": len(completed)}
    for key in ("best_val_rmse", "test_rmse", "test_phm_score"):
        values = np.asarray([row[key] for row in completed], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "std_population": float(values.std(ddof=0)) if len(values) else None,
            "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    return summary


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_results(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
    document = {
        "dataset": "FD001",
        "seeds": SEEDS,
        "new_seeds": NEW_SEEDS,
        "training": TRAINING,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")

    fields = [
        "seed",
        "status",
        "best_val_rmse",
        "test_rmse",
        "test_phm_score",
        "elapsed_seconds",
        "checkpoint",
        "predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_seed: dict[int, dict[str, Any]] = {}

    for seed in SEEDS:
        completed = load_completed(seed)
        if completed is not None:
            rows_by_seed[seed] = completed
    save_results(rows_by_seed)

    for seed in NEW_SEEDS:
        if seed in rows_by_seed:
            print(f"seed={seed} already completed; skipping")
            continue

        started = time.monotonic()
        checkpoint = checkpoint_path(seed)
        log_path = CACHE_DIR / f"adapter_wide_deep_seed{seed}.training.log"
        print(f"seed={seed} starting; log={log_path.relative_to(PROJECT_ROOT)}")
        try:
            with log_path.open("a", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)):
                train_result = train_adapter(
                    CACHE_DIR,
                    checkpoint,
                    seed=seed,
                    **TRAINING,
                )
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
                "best_val_rmse": float(train_result["best_val_rmse"]),
                "test_rmse": float(test_result["rmse"]),
                "test_phm_score": float(test_result["score"]),
                "elapsed_seconds": round(elapsed, 3),
                "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
                "predictions": str(prediction_path(seed).relative_to(PROJECT_ROOT)),
            }
            save_results(rows_by_seed)
            print(
                f"seed={seed} completed in {elapsed:.1f}s; "
                f"val_rmse={train_result['best_val_rmse']:.5f} "
                f"test_rmse={test_result['rmse']:.5f} score={test_result['score']:.5f}"
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
            save_results(rows_by_seed)
            raise

    save_results(rows_by_seed)
    final = aggregate(list(rows_by_seed.values()))
    if final["count"] != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} completed runs, found {final['count']}")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
