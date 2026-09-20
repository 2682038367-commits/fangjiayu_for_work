#!/usr/bin/env python3
"""Select an FD001 Chronos representation by validation RMSE, then test only the winner."""

from __future__ import annotations

import csv
import json
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from rul_chronos.training import evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "artifacts" / "FD001" / "representation_ablation"
REPRESENTATIONS = ("last_valid_patch", "valid_patch_mean", "reg_token")
JSON_PATH = ROOT / "representation_results.json"
CSV_PATH = ROOT / "representation_results.csv"
SEED = 42
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


def checkpoint_path(representation: str) -> Path:
    return ROOT / representation / "adapter_wide_deep_seed42.pt"


def load_or_train(representation: str) -> dict[str, Any]:
    cache = ROOT / representation
    checkpoint = checkpoint_path(representation)
    log = cache / "adapter_wide_deep_seed42.training.log"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return {
            "representation": representation,
            "best_epoch": int(saved["best_epoch"]),
            "completed_epochs": int(saved["completed_epochs"]),
            "best_val_rmse": float(saved["best_val_rmse"]),
            "elapsed_seconds": None,
            "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        }
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output, redirect_stdout(output):
        result = train_adapter(cache, checkpoint, seed=SEED, **TRAINING)
    return {
        "representation": representation,
        "best_epoch": int(result["best_epoch"]),
        "completed_epochs": int(result["completed_epochs"]),
        "best_val_rmse": float(result["best_val_rmse"]),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
    }


def save(rows: list[dict[str, Any]], selected: str, test_result: dict[str, float]) -> None:
    document = {
        "dataset": "FD001",
        "seed": SEED,
        "selection_rule": "lowest all-prefix validation RMSE; test is evaluated only for the selected representation",
        "representations": list(REPRESENTATIONS),
        "fixed_training": TRAINING,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "selected_representation": selected,
        "selected_test_result": test_result,
    }
    JSON_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    fields = [
        "representation", "best_epoch", "completed_epochs", "best_val_rmse",
        "test_evaluated", "test_rmse", "test_phm_score", "elapsed_seconds", "checkpoint", "predictions",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    if not (ROOT / "manifest.json").exists():
        raise RuntimeError("run extract_fd001_representation_variants.py first")
    rows = []
    for representation in REPRESENTATIONS:
        print(f"training representation={representation}")
        row = load_or_train(representation)
        row.update({"test_evaluated": False, "test_rmse": None, "test_phm_score": None, "predictions": None})
        rows.append(row)
        print(
            f"validation representation={representation} best_epoch={row['best_epoch']} "
            f"rmse={row['best_val_rmse']:.6f}"
        )
    selected_row = min(rows, key=lambda row: row["best_val_rmse"])
    selected = selected_row["representation"]
    print(f"selected by validation only: {selected}")
    test_result = evaluate_adapter(ROOT / selected, checkpoint_path(selected), TRAINING["batch_size"], TRAINING["device_name"])
    selected_row["test_evaluated"] = True
    selected_row["test_rmse"] = float(test_result["rmse"])
    selected_row["test_phm_score"] = float(test_result["score"])
    selected_row["predictions"] = str(checkpoint_path(selected).with_suffix(".predictions.json").relative_to(PROJECT_ROOT))
    save(rows, selected, test_result)
    print(json.dumps({"selected_representation": selected, **test_result}, indent=2))


if __name__ == "__main__":
    main()
