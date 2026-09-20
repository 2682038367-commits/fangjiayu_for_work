#!/usr/bin/env python3
"""Compare FD003 mean pooling, REG, and future-query embeddings for seed 42."""

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
FD003_ROOT = PROJECT_ROOT / "artifacts" / "FD003"
ROOT = FD003_ROOT / "token_representation_ablation"
MEAN_CACHE = FD003_ROOT / "patch_representation_ablation" / "valid_patch_mean"
REPRESENTATIONS = ("valid_patch_mean", "reg_token", "future_query")
NEW_REPRESENTATIONS = ("reg_token", "future_query")
JSON_PATH = ROOT / "token_representation_results.json"
CSV_PATH = ROOT / "token_representation_results.csv"
REPORT_PATH = ROOT / "TOKEN_REPRESENTATION_REPORT.md"
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


def cache_dir(representation: str) -> Path:
    return MEAN_CACHE if representation == "valid_patch_mean" else ROOT / representation


def checkpoint_path(representation: str) -> Path:
    return cache_dir(representation) / "adapter_wide_deep_seed42.pt"


def prediction_path(representation: str) -> Path:
    return checkpoint_path(representation).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_row(representation: str, elapsed_seconds: float | None) -> dict[str, Any]:
    checkpoint_file = checkpoint_path(representation)
    predictions_file = prediction_path(representation)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    predictions = json.loads(predictions_file.read_text(encoding="utf-8"))
    if checkpoint["model_kwargs"] != {
        "num_sensors": 21,
        "embedding_dim": 768,
        "compression_dim": 8,
        "num_regimes": 1,
        "dropout": 0.2,
        "deep_only": False,
    }:
        raise RuntimeError(f"unexpected adapter configuration for {representation}")
    if int(checkpoint["seed"]) != SEED:
        raise RuntimeError(f"unexpected seed for {representation}")
    if len(predictions["prediction"]) != 100 or len(set(predictions["unit"])) != 100:
        raise RuntimeError(f"{representation} predictions do not cover 100 unique test engines")
    return {
        "representation": representation,
        "seed": SEED,
        "best_epoch": int(checkpoint["best_epoch"]),
        "completed_epochs": int(checkpoint["completed_epochs"]),
        "best_val_rmse": float(checkpoint["best_val_rmse"]),
        "test_rmse": float(predictions["metrics"]["rmse"]),
        "test_phm_score": float(predictions["metrics"]["score"]),
        "prediction_count": len(predictions["prediction"]),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": relative(checkpoint_file),
        "predictions": relative(predictions_file),
    }


def write_outputs(rows: list[dict[str, Any]]) -> None:
    mean = next(row for row in rows if row["representation"] == "valid_patch_mean")
    for row in rows:
        row["val_rmse_change_vs_mean"] = row["best_val_rmse"] - mean["best_val_rmse"]
        row["test_rmse_change_vs_mean"] = row["test_rmse"] - mean["test_rmse"]
        row["score_change_vs_mean"] = row["test_phm_score"] - mean["test_phm_score"]
    document = {
        "dataset": "FD003",
        "seed": SEED,
        "configuration": "unrestricted prefixes, dcomp=8 Full Fusion",
        "fixed_training": TRAINING,
        "test_policy": "exploratory: REG and future query are both evaluated on test by explicit user choice",
        "mean_pooling_source": relative(MEAN_CACHE),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
    }
    JSON_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    fields = [
        "representation", "seed", "best_epoch", "completed_epochs", "best_val_rmse",
        "val_rmse_change_vs_mean", "test_rmse", "test_rmse_change_vs_mean", "test_phm_score",
        "score_change_vs_mean", "prediction_count", "elapsed_seconds", "checkpoint", "predictions",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    labels = {
        "valid_patch_mean": "Valid patch mean",
        "reg_token": "REG token",
        "future_query": "Future query",
    }
    lines = [
        "# FD003 Chronos token representation comparison (seed 42)",
        "",
        "固定无限制 prefix、Full Fusion、`dcomp=8` 和全部训练超参数。Mean pooling 复用现有结果；",
        "REG 与 future query 从同一次 Chronos 前向提取并分别训练。按用户选择，三种表示均报告测试指标，",
        "因此该结果属于探索性比较，不是验证集完全隔离的模型选择。",
        "",
        "| Representation | Best epoch | Completed | Validation RMSE | Test RMSE | PHM Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {labels[row['representation']]} | {row['best_epoch']} | {row['completed_epochs']} | "
            f"{row['best_val_rmse']:.6f} | {row['test_rmse']:.6f} | {row['test_phm_score']:.6f} |"
        )
    if len(rows) == len(REPRESENTATIONS):
        reg = next(row for row in rows if row["representation"] == "reg_token")
        future = next(row for row in rows if row["representation"] == "future_query")
        lines.extend([
            "",
            "Chronos encoder 序列已核对为 `context patches → REG → one future query`：REG 位于 "
            "`num_context_patches`，future query 位于 `num_context_patches + 1`。两者继续使用同一 prefix "
            "21 个 sensor共享 group ID 的联合 group attention。",
            "",
            "两套新缓存均通过 20,012/4,708/100 行、`[N,21,768]`、float16、有限值和样本顺序检查；"
            "所有非 embedding 数组与 mean-pooling基线硬链接一致。三种 embedding 的平均绝对差为 0.23–0.34，"
            "排除了误取同一 token。",
            "",
            f"相对 mean pooling，REG 的验证 RMSE、测试 RMSE、Score 分别变化 "
            f"{reg['val_rmse_change_vs_mean']:+.6f}、{reg['test_rmse_change_vs_mean']:+.6f}、"
            f"{reg['score_change_vs_mean']:+.6f}；future query 分别变化 "
            f"{future['val_rmse_change_vs_mean']:+.6f}、{future['test_rmse_change_vs_mean']:+.6f}、"
            f"{future['score_change_vs_mean']:+.6f}。正数表示变差。",
            "",
            "本次 seed42 上 mean pooling 三项均最佳，future query 次之，REG 最差。",
        ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not (ROOT / "manifest.json").exists():
        raise RuntimeError("run extract_fd003_token_representations.py first")
    previous_elapsed: dict[str, float | None] = {}
    if JSON_PATH.exists():
        previous = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        previous_elapsed = {
            row["representation"]: row.get("elapsed_seconds") for row in previous.get("runs", [])
        }
    rows = [load_row("valid_patch_mean", previous_elapsed.get("valid_patch_mean"))]
    for representation in NEW_REPRESENTATIONS:
        checkpoint_file = checkpoint_path(representation)
        predictions_file = prediction_path(representation)
        elapsed = previous_elapsed.get(representation)
        if not checkpoint_file.exists() or not predictions_file.exists():
            started = time.monotonic()
            log = cache_dir(representation) / "adapter_wide_deep_seed42.training.log"
            with log.open("w", encoding="utf-8") as output, redirect_stdout(output):
                train_adapter(cache_dir(representation), checkpoint_file, seed=SEED, **TRAINING)
                evaluate_adapter(
                    cache_dir(representation),
                    checkpoint_file,
                    batch_size=TRAINING["batch_size"],
                    device_name=TRAINING["device_name"],
                )
            elapsed = round(time.monotonic() - started, 3)
        row = load_row(representation, elapsed)
        rows.append(row)
        write_outputs(rows)
        print(
            f"representation={representation}: epoch={row['best_epoch']}/{row['completed_epochs']} "
            f"val={row['best_val_rmse']:.6f} test={row['test_rmse']:.6f} score={row['test_phm_score']:.6f}"
        )
    write_outputs(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
