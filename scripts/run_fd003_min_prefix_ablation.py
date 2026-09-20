#!/usr/bin/env python3
"""Compare unrestricted, >=16, and >=32 prefixes for FD003 mean-patch Full Fusion."""

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

from rul_chronos.cache import EmbeddingCacheDataset
from rul_chronos.metrics import rmse
from rul_chronos.model import WideDeepRULAdapter
from rul_chronos.training import _cache_on_device, _predict_cached, evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD003" / "patch_representation_ablation" / "valid_patch_mean"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "FD003" / "patch_representation_ablation" / "prefix_min_ablation"
JSON_PATH = OUTPUT_DIR / "prefix_min_results.json"
CSV_PATH = OUTPUT_DIR / "prefix_min_results.csv"
REPORT_PATH = OUTPUT_DIR / "PREFIX_MIN_REPORT.md"
SEED = 42
MIN_PREFIXES = (1, 16, 32)
EVALUATION_THRESHOLDS = (1, 16, 32)
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


def checkpoint_path(min_prefix: int) -> Path:
    if min_prefix == 1:
        return CACHE_DIR / "adapter_wide_deep_seed42.pt"
    return OUTPUT_DIR / f"adapter_wide_deep_seed42_minprefix{min_prefix}.pt"


def prediction_path(min_prefix: int) -> Path:
    return checkpoint_path(min_prefix).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def split_size(split: str, min_prefix: int) -> int:
    cycles = np.load(CACHE_DIR / f"{split}_cycles.npy", mmap_mode="r")
    return int(np.count_nonzero(cycles >= min_prefix))


@torch.inference_mode()
def validation_rmse(checkpoint_file: Path, min_prefix: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    model = WideDeepRULAdapter(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    dataset = EmbeddingCacheDataset(CACHE_DIR, "val", min_prefix_length=min_prefix)
    tensors = _cache_on_device(dataset, device)
    target, prediction = _predict_cached(model, tensors, TRAINING["batch_size"])
    return float(rmse(target, prediction))


def train_or_load(min_prefix: int) -> dict[str, Any]:
    checkpoint_file = checkpoint_path(min_prefix)
    predictions_file = prediction_path(min_prefix)
    elapsed: float | None = None
    if min_prefix > 1 and (not checkpoint_file.exists() or not predictions_file.exists()):
        started = time.monotonic()
        log = OUTPUT_DIR / f"adapter_wide_deep_seed42_minprefix{min_prefix}.training.log"
        with log.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_adapter(
                CACHE_DIR,
                checkpoint_file,
                seed=SEED,
                min_prefix_length=min_prefix,
                **TRAINING,
            )
            evaluate_adapter(
                CACHE_DIR,
                checkpoint_file,
                batch_size=TRAINING["batch_size"],
                device_name=TRAINING["device_name"],
            )
        elapsed = round(time.monotonic() - started, 3)
    if not checkpoint_file.exists() or not predictions_file.exists():
        raise RuntimeError(f"missing checkpoint or predictions for min_prefix={min_prefix}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    predictions = json.loads(predictions_file.read_text(encoding="utf-8"))
    if len(predictions["prediction"]) != 100 or len(set(predictions["unit"])) != 100:
        raise RuntimeError(f"min_prefix={min_prefix} does not have 100 unique test predictions")
    configured_min = int(checkpoint.get("min_prefix_length", 1))
    if configured_min != min_prefix:
        raise RuntimeError(f"checkpoint prefix threshold mismatch: {configured_min} != {min_prefix}")
    row: dict[str, Any] = {
        "min_prefix_length": min_prefix,
        "seed": SEED,
        "train_samples": split_size("train", min_prefix),
        "validation_selection_samples": split_size("val", min_prefix),
        "test_samples": 100,
        "best_epoch": int(checkpoint["best_epoch"]),
        "completed_epochs": int(checkpoint["completed_epochs"]),
        "best_val_rmse": float(checkpoint["best_val_rmse"]),
        "test_rmse": float(predictions["metrics"]["rmse"]),
        "test_phm_score": float(predictions["metrics"]["score"]),
        "elapsed_seconds": elapsed,
        "checkpoint": relative(checkpoint_file),
        "predictions": relative(predictions_file),
    }
    for threshold in EVALUATION_THRESHOLDS:
        row[f"val_ge{threshold}_rmse"] = validation_rmse(checkpoint_file, threshold)
    selection_key = f"val_ge{min_prefix}_rmse"
    if not np.isclose(row[selection_key], row["best_val_rmse"], rtol=0.0, atol=1e-5):
        raise RuntimeError(
            f"min_prefix={min_prefix} selection RMSE mismatch: "
            f"{row[selection_key]} vs {row['best_val_rmse']}"
        )
    return row


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save(rows: list[dict[str, Any]]) -> None:
    baseline = rows[0]
    for row in rows:
        row["test_rmse_change_vs_unrestricted"] = row["test_rmse"] - baseline["test_rmse"]
        row["test_score_change_vs_unrestricted"] = row["test_phm_score"] - baseline["test_phm_score"]
        row["common_val_ge32_change_vs_unrestricted"] = row["val_ge32_rmse"] - baseline["val_ge32_rmse"]
    document = {
        "dataset": "FD003",
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion",
        "seed": SEED,
        "definition": "min_prefix_length filters train and validation prefixes by cycle; official test endpoints are unchanged",
        "fixed_training": TRAINING,
        "common_validation_comparison": "all checkpoints evaluated on the same validation prefixes with cycle >= 32",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")
    fields = [
        "min_prefix_length", "seed", "train_samples", "validation_selection_samples", "test_samples",
        "best_epoch", "completed_epochs", "best_val_rmse", "val_ge1_rmse", "val_ge16_rmse", "val_ge32_rmse",
        "common_val_ge32_change_vs_unrestricted", "test_rmse", "test_rmse_change_vs_unrestricted",
        "test_phm_score", "test_score_change_vs_unrestricted", "elapsed_seconds", "checkpoint", "predictions",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    lines = [
        "# FD003 最小 prefix 长度实验（seed 42）",
        "",
        "固定有效 patch 均值、Full Fusion、`dcomp=8` 和全部训练超参数。最小长度过滤同时应用于训练和验证；",
        "官方测试的 100 个发动机终点保持不变。由于各自 early-stopping 验证集不同，另外统一报告 `cycle≥32` 验证 RMSE。",
        "",
        "| Min prefix | Train N | Selection val N | Best epoch | Own val RMSE | Common val ≥32 RMSE | Test RMSE | Score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['min_prefix_length']} | {row['train_samples']} | {row['validation_selection_samples']} | "
            f"{row['best_epoch']} | {row['best_val_rmse']:.6f} | {row['val_ge32_rmse']:.6f} | "
            f"{row['test_rmse']:.6f} | {row['test_phm_score']:.6f} |"
        )
    if len(rows) == len(MIN_PREFIXES):
        min16, min32 = rows[1], rows[2]
        lines.extend([
            "",
            "各组的 Own val RMSE 来自不同大小的验证集，不能直接横向排序。共同 `cycle≥32` 口径下，",
            f"min=16 相对无限制变化 {min16['common_val_ge32_change_vs_unrestricted']:+.6f}，"
            f"min=32 变化 {min32['common_val_ge32_change_vs_unrestricted']:+.6f}。",
            "",
            f"测试集上，min=16 的 RMSE 变化 {min16['test_rmse_change_vs_unrestricted']:+.6f}、"
            f"Score 变化 {min16['test_score_change_vs_unrestricted']:+.6f}；"
            f"min=32 的 RMSE 变化 {min32['test_rmse_change_vs_unrestricted']:+.6f}、"
            f"Score 变化 {min32['test_score_change_vs_unrestricted']:+.6f}。负数表示改善。",
            "",
            "单 seed 的验证与测试排序不一致，不能据此确定最优最小 prefix。过滤模型在被排除的早期前缀上属于外推，",
            f"全验证前缀 RMSE 分别为 {rows[0]['val_ge1_rmse']:.6f}、{min16['val_ge1_rmse']:.6f}、"
            f"{min32['val_ge1_rmse']:.6f}。",
        ])
    atomic_write(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for min_prefix in MIN_PREFIXES:
        print(f"processing min_prefix={min_prefix}")
        row = train_or_load(min_prefix)
        rows.append(row)
        save(rows)
        print(
            f"min_prefix={min_prefix}: epoch={row['best_epoch']}/{row['completed_epochs']} "
            f"own_val={row['best_val_rmse']:.6f} common32={row['val_ge32_rmse']:.6f} "
            f"test={row['test_rmse']:.6f} score={row['test_phm_score']:.6f}"
        )
    save(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
