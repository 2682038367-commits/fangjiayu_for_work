#!/usr/bin/env python3
"""Train and verify the standard FD002 mean-patch Full Fusion seed-42 run."""

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
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD002"
CHECKPOINT = CACHE_DIR / "adapter_wide_deep_seed42.pt"
LOG_PATH = CACHE_DIR / "adapter_wide_deep_seed42.training.log"
JSON_PATH = CACHE_DIR / "run_summary_seed42.json"
CSV_PATH = CACHE_DIR / "run_summary_seed42.csv"
REPORT_PATH = CACHE_DIR / "RUN_REPORT_SEED42.md"
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
    "min_prefix_length": 1,
}


def verify_predictions(metrics: dict[str, float]) -> dict[str, object]:
    path = CHECKPOINT.with_suffix(".predictions.json")
    prediction_data = json.loads(path.read_text(encoding="utf-8"))
    units = np.asarray(prediction_data["unit"], dtype=np.int64)
    cycles = np.asarray(prediction_data["cycle"], dtype=np.int64)
    target = np.asarray(prediction_data["target"], dtype=np.float64)
    prediction = np.asarray(prediction_data["prediction"], dtype=np.float64)
    if len(units) != 259 or len(np.unique(units)) != 259:
        raise RuntimeError("FD002 predictions do not cover 259 unique test engines")
    if not all(len(values) == 259 for values in (cycles, target, prediction)):
        raise RuntimeError("FD002 prediction arrays have inconsistent lengths")
    recomputed = {"rmse": rmse(target, prediction), "score": phm_score(target, prediction)}
    for key in recomputed:
        if not np.isclose(recomputed[key], metrics[key], rtol=0.0, atol=1e-10):
            raise RuntimeError(f"independent {key} does not match stored result")
    return {
        "prediction_count": int(len(prediction)),
        "unique_test_engines": int(len(np.unique(units))),
        "all_finite": bool(np.isfinite(prediction).all()),
        "independent_rmse": float(recomputed["rmse"]),
        "independent_score": float(recomputed["score"]),
    }


def validation_metrics(saved: dict[str, object]) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideDeepRULAdapter(**saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    validation = EmbeddingCacheDataset(CACHE_DIR, "val")
    target, prediction = predict(
        model,
        DataLoader(validation, batch_size=TRAINING["batch_size"], shuffle=False),
        device,
    )
    units = np.asarray(validation.units, dtype=np.int64)
    cycles = np.asarray(validation.cycles, dtype=np.int64)
    endpoint_indices = np.asarray(
        [np.flatnonzero(units == unit)[np.argmax(cycles[units == unit])] for unit in np.unique(units)],
        dtype=np.int64,
    )
    all_prefix_rmse = rmse(target, prediction)
    if not np.isclose(all_prefix_rmse, float(saved["best_val_rmse"]), rtol=0.0, atol=1e-5):
        raise RuntimeError("recomputed all-prefix validation RMSE differs from checkpoint")
    return {
        "val_all_prefix_rmse": float(all_prefix_rmse),
        "val_endpoint_rmse": float(rmse(target[endpoint_indices], prediction[endpoint_indices])),
    }


def main() -> None:
    if not (CACHE_DIR / "manifest.json").exists():
        raise RuntimeError("run scripts/extract_fd002_mean_patch.py first")
    started = time.monotonic()
    if not CHECKPOINT.exists():
        with LOG_PATH.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_adapter(CACHE_DIR, CHECKPOINT, seed=SEED, **TRAINING)
        elapsed_seconds: float | None = round(time.monotonic() - started, 3)
    else:
        elapsed_seconds = None
        if JSON_PATH.exists():
            previous = json.loads(JSON_PATH.read_text(encoding="utf-8"))
            elapsed_seconds = previous.get("run", {}).get("elapsed_seconds")
        print(f"reusing existing checkpoint: {CHECKPOINT.relative_to(PROJECT_ROOT)}")

    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    expected_kwargs = {
        "compression_dim": 8,
        "dropout": 0.2,
        "deep_only": False,
        "num_regimes": 6,
        "num_sensors": 21,
        "embedding_dim": 768,
    }
    for key, expected in expected_kwargs.items():
        if saved["model_kwargs"][key] != expected:
            raise RuntimeError(f"checkpoint {key} differs: {saved['model_kwargs'][key]} != {expected}")
    if int(saved["seed"]) != SEED or int(saved["min_prefix_length"]) != 1:
        raise RuntimeError("checkpoint seed or prefix restriction differs from requested run")

    metrics = evaluate_adapter(CACHE_DIR, CHECKPOINT, TRAINING["batch_size"], TRAINING["device_name"])
    verification = verify_predictions(metrics)
    validation = validation_metrics(saved)
    row = {
        "dataset": "FD002",
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
        "test_samples": 259,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "predictions": str(CHECKPOINT.with_suffix(".predictions.json").relative_to(PROJECT_ROOT)),
    }
    document = {
        "experiment": "FD002 standard Full Fusion end-to-end check",
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
                "# FD002 Full Fusion 端到端检查（seed 42）",
                "",
                "固定 `valid_patch_mean`、21-sensor 联合 group attention、无限制 prefix、",
                "`dcomp=8`、AdamW、lr=1e-4、weight decay=1e-3、batch size 1024、",
                "dropout 0.2、max 150 epochs、patience 15。",
                "",
                "| Best epoch | Completed | Val all-prefix RMSE | Val endpoint RMSE | Test RMSE | PHM Score | Test engines |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {row['best_epoch']} | {row['completed_epochs']} | {row['val_all_prefix_rmse']:.6f} | "
                f"{row['val_endpoint_rmse']:.6f} | {row['test_rmse']:.6f} | "
                f"{row['test_phm_score']:.6f} | 259 |",
                "",
                "Early stopping 使用验证集全部历史前缀；endpoint 指标只取52台验证发动机的最后时刻。",
                "验证轨迹完整运行至失效，因此其 endpoint 标签均为0；官方测试轨迹提前截断，",
                "所以 val endpoint RMSE 不能与 test RMSE 直接比较。",
                "测试指标已由预测文件独立复算，且覆盖 259 台唯一发动机。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
