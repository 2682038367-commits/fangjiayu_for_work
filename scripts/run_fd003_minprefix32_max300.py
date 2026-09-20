#!/usr/bin/env python3
"""Retrain FD003 seed42 mean-patch Full Fusion with min-prefix 32 and max 300 epochs."""

from __future__ import annotations

import json
import re
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rul_chronos.cache import EmbeddingCacheDataset
from rul_chronos.metrics import rmse
from rul_chronos.model import WideDeepRULAdapter
from rul_chronos.training import _cache_on_device, _predict_cached, evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD003" / "patch_representation_ablation" / "valid_patch_mean"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "FD003" / "patch_representation_ablation" / "prefix_min_ablation"
BASELINE_RESULTS = OUTPUT_DIR / "prefix_min_results.json"
CHECKPOINT = OUTPUT_DIR / "adapter_wide_deep_seed42_minprefix32_max300.pt"
PREDICTIONS = CHECKPOINT.with_suffix(".predictions.json")
LOG_PATH = OUTPUT_DIR / "adapter_wide_deep_seed42_minprefix32_max300.training.log"
JSON_PATH = OUTPUT_DIR / "minprefix32_max300_results.json"
REPORT_PATH = OUTPUT_DIR / "MINPREFIX32_MAX300_REPORT.md"
SEED = 42
MIN_PREFIX = 32
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


@torch.inference_mode()
def validation_rmse(checkpoint_path: Path, threshold: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = WideDeepRULAdapter(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    dataset = EmbeddingCacheDataset(CACHE_DIR, "val", min_prefix_length=threshold)
    tensors = _cache_on_device(dataset, device)
    target, prediction = _predict_cached(model, tensors, TRAINING["batch_size"])
    return float(rmse(target, prediction))


def baseline() -> dict:
    document = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    return next(row for row in document["runs"] if int(row["min_prefix_length"]) == MIN_PREFIX)


def epoch150_val() -> float:
    pattern = re.compile(r"^epoch=150 .* val_rmse=([0-9.]+)$")
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise RuntimeError("max300 training log does not contain epoch 150")


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    old = baseline()
    elapsed = None
    if not CHECKPOINT.exists() or not PREDICTIONS.exists():
        started = time.monotonic()
        with LOG_PATH.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_adapter(
                CACHE_DIR,
                CHECKPOINT,
                seed=SEED,
                min_prefix_length=MIN_PREFIX,
                **TRAINING,
            )
            evaluate_adapter(
                CACHE_DIR,
                CHECKPOINT,
                batch_size=TRAINING["batch_size"],
                device_name=TRAINING["device_name"],
            )
        elapsed = round(time.monotonic() - started, 3)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if int(checkpoint.get("min_prefix_length", 1)) != MIN_PREFIX:
        raise RuntimeError("max300 checkpoint has the wrong minimum prefix length")
    if len(predictions["prediction"]) != 100 or len(set(predictions["unit"])) != 100:
        raise RuntimeError("max300 predictions do not cover 100 unique test engines")

    new_val = validation_rmse(CHECKPOINT, MIN_PREFIX)
    if not np.isclose(new_val, checkpoint["best_val_rmse"], rtol=0.0, atol=1e-5):
        raise RuntimeError("recomputed max300 validation RMSE differs from its checkpoint")
    logged_epoch150_val = epoch150_val()
    epoch150_matches = bool(np.isclose(logged_epoch150_val, old["best_val_rmse"], rtol=0.0, atol=1e-5))
    result = {
        "dataset": "FD003",
        "seed": SEED,
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion, min_prefix_length=32",
        "change": "max_epochs 150 -> 300; all other settings unchanged; retrained from epoch 1",
        "training": TRAINING,
        "max150": {
            "best_epoch": int(old["best_epoch"]),
            "completed_epochs": int(old["completed_epochs"]),
            "validation_rmse_ge32": float(old["val_ge32_rmse"]),
            "test_rmse": float(old["test_rmse"]),
            "test_phm_score": float(old["test_phm_score"]),
            "checkpoint": old["checkpoint"],
            "predictions": old["predictions"],
        },
        "max300": {
            "best_epoch": int(checkpoint["best_epoch"]),
            "completed_epochs": int(checkpoint["completed_epochs"]),
            "validation_rmse_ge32": new_val,
            "validation_rmse_all_prefixes": validation_rmse(CHECKPOINT, 1),
            "test_rmse": float(predictions["metrics"]["rmse"]),
            "test_phm_score": float(predictions["metrics"]["score"]),
            "elapsed_seconds": elapsed,
            "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
            "predictions": str(PREDICTIONS.relative_to(PROJECT_ROOT)),
        },
        "changes_max300_minus_max150": {
            "validation_rmse_ge32": new_val - float(old["val_ge32_rmse"]),
            "test_rmse": float(predictions["metrics"]["rmse"]) - float(old["test_rmse"]),
            "test_phm_score": float(predictions["metrics"]["score"]) - float(old["test_phm_score"]),
        },
        "trajectory_check": {
            "max300_epoch150_val_rmse": logged_epoch150_val,
            "max150_best_val_rmse": float(old["best_val_rmse"]),
            "matches_within_1e-5": epoch150_matches,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not epoch150_matches:
        raise RuntimeError("max300 run does not reproduce the max150 trajectory through epoch 150")
    atomic_write(JSON_PATH, json.dumps(result, indent=2) + "\n")

    new = result["max300"]
    changes = result["changes_max300_minus_max150"]
    lines = [
        "# FD003 min-prefix 32：max150 与 max300",
        "",
        "固定 seed 42、有效 patch 均值、Full Fusion、`dcomp=8`、patience 15 和其余超参数；",
        "仅将最大训练轮次从 150 改为 300，并从 epoch 1 确定性重训。",
        "",
        "| 上限 | Best epoch | Completed | Val RMSE (cycle≥32) | Test RMSE | Score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| 150 | {old['best_epoch']} | {old['completed_epochs']} | {old['val_ge32_rmse']:.6f} | "
        f"{old['test_rmse']:.6f} | {old['test_phm_score']:.6f} |",
        f"| 300 | {new['best_epoch']} | {new['completed_epochs']} | {new['validation_rmse_ge32']:.6f} | "
        f"{new['test_rmse']:.6f} | {new['test_phm_score']:.6f} |",
        "",
        f"max300−max150：验证 RMSE {changes['validation_rmse_ge32']:+.6f}，"
        f"测试 RMSE {changes['test_rmse']:+.6f}，Score {changes['test_phm_score']:+.6f}。负数表示改善。",
        "",
        f"轨迹核验：max300 在 epoch 150 的验证 RMSE 为 {logged_epoch150_val:.5f}，"
        f"与 max150 checkpoint 的 {old['best_val_rmse']:.6f} 在 1e-5 内一致。",
    ]
    atomic_write(REPORT_PATH, "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
