#!/usr/bin/env python3
"""Train and verify the fixed FD004 mean-patch Full Fusion seed-42 run."""

from __future__ import annotations

import csv
import json
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rul_chronos.cache import EmbeddingCacheDataset
from rul_chronos.metrics import phm_score, rmse
from rul_chronos.model import WideDeepRULAdapter
from rul_chronos.training import evaluate_adapter, predict, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD004"
CHECKPOINT = CACHE_DIR / "adapter_wide_deep_seed42.pt"
LOG_PATH = CACHE_DIR / "adapter_wide_deep_seed42.training.log"
JSON_PATH = CACHE_DIR / "run_summary_seed42.json"
CSV_PATH = CACHE_DIR / "run_summary_seed42.csv"
REPORT_PATH = CACHE_DIR / "RUN_REPORT_SEED42.md"
SEED = 42
TEST_ENGINES = 248
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


def validation_metrics(saved: dict[str, object]) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideDeepRULAdapter(**saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    validation = EmbeddingCacheDataset(CACHE_DIR, "val")
    target, prediction = predict(model, DataLoader(validation, batch_size=1024, shuffle=False), device)
    units = np.asarray(validation.units, dtype=np.int64)
    endpoint_indices = np.asarray([np.flatnonzero(units == unit)[-1] for unit in np.unique(units)])
    all_prefix = rmse(target, prediction)
    if not np.isclose(all_prefix, float(saved["best_val_rmse"]), rtol=0.0, atol=1e-5):
        raise RuntimeError("recomputed all-prefix validation RMSE differs from checkpoint")
    return {
        "val_all_prefix_rmse": float(all_prefix),
        "val_endpoint_rmse": float(rmse(target[endpoint_indices], prediction[endpoint_indices])),
    }


def verify_predictions(metrics: dict[str, float]) -> dict[str, object]:
    path = CHECKPOINT.with_suffix(".predictions.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    units = np.asarray(data["unit"], dtype=np.int64)
    target = np.asarray(data["target"], dtype=np.float64)
    prediction = np.asarray(data["prediction"], dtype=np.float64)
    if len(prediction) != TEST_ENGINES or len(np.unique(units)) != TEST_ENGINES:
        raise RuntimeError("FD004 predictions do not cover 248 unique test engines")
    if not np.isfinite(prediction).all():
        raise RuntimeError("FD004 predictions contain non-finite values")
    recomputed = {"rmse": rmse(target, prediction), "score": phm_score(target, prediction)}
    for key, value in recomputed.items():
        if not np.isclose(value, metrics[key], rtol=0.0, atol=1e-10):
            raise RuntimeError(f"independent {key} differs from stored result")
    return {
        "prediction_count": int(len(prediction)),
        "unique_test_engines": int(len(np.unique(units))),
        "all_finite": True,
        "independent_rmse": float(recomputed["rmse"]),
        "independent_score": float(recomputed["score"]),
    }


def main() -> None:
    if not (CACHE_DIR / "manifest.json").exists():
        raise RuntimeError("run scripts/extract_fd004_mean_patch.py first")
    started = time.monotonic()
    if not CHECKPOINT.exists():
        with LOG_PATH.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_adapter(CACHE_DIR, CHECKPOINT, seed=SEED, **TRAINING)
        elapsed_seconds: float | None = round(time.monotonic() - started, 3)
    else:
        elapsed_seconds = None
        if JSON_PATH.exists():
            elapsed_seconds = json.loads(JSON_PATH.read_text(encoding="utf-8")).get("run", {}).get("elapsed_seconds")
        print(f"reusing existing checkpoint: {CHECKPOINT.relative_to(PROJECT_ROOT)}")

    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected_kwargs = {
        "num_sensors": 21,
        "embedding_dim": 768,
        "compression_dim": 8,
        "num_regimes": 6,
        "dropout": 0.2,
        "deep_only": False,
    }
    if saved["model_kwargs"] != expected_kwargs:
        raise RuntimeError("FD004 checkpoint model configuration differs")
    if int(saved["seed"]) != SEED or int(saved["min_prefix_length"]) != 1:
        raise RuntimeError("FD004 checkpoint seed or prefix setting differs")

    metrics = evaluate_adapter(CACHE_DIR, CHECKPOINT, 1024, "auto")
    validation = validation_metrics(saved)
    verification = verify_predictions(metrics)
    row = {
        "dataset": "FD004",
        "representation": "valid_patch_mean",
        "architecture": "full_fusion",
        "seed": SEED,
        "best_epoch": int(saved["best_epoch"]),
        "completed_epochs": int(saved["completed_epochs"]),
        "best_val_rmse": float(saved["best_val_rmse"]),
        **validation,
        "test_rmse": float(metrics["rmse"]),
        "test_phm_score": float(metrics["score"]),
        "train_samples": int(saved["train_samples"]),
        "validation_samples": int(saved["validation_samples"]),
        "test_samples": TEST_ENGINES,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "predictions": str(CHECKPOINT.with_suffix(".predictions.json").relative_to(PROJECT_ROOT)),
    }
    document = {
        "experiment": "FD004 fixed mean-patch Full Fusion end-to-end check",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "fixed_training": TRAINING,
        "run": row,
        "verification": verification,
    }
    JSON_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# FD004 Full Fusion 端到端检查（seed 42）",
                "",
                "固定 `valid_patch_mean`、21-sensor联合 group attention、无限制 prefix、`dcomp=8`、",
                "AdamW、lr=1e-4、weight decay=1e-3、batch size 1024、dropout 0.2、",
                "max 150 epochs、patience 15。",
                "",
                "| Best epoch | Completed | Val all-prefix | Val endpoint | Test RMSE | PHM Score | Test engines |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {row['best_epoch']} | {row['completed_epochs']} | {row['val_all_prefix_rmse']:.6f} | "
                f"{row['val_endpoint_rmse']:.6f} | {row['test_rmse']:.6f} | "
                f"{row['test_phm_score']:.6f} | {TEST_ENGINES} |",
                "",
                "Early stopping 使用验证集全部历史前缀。验证轨迹 endpoint 标签均为0，",
                "不能与提前截断的官方测试终点 RMSE 直接比较。测试指标已独立复算。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
