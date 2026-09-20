#!/usr/bin/env python3
"""Run a fixed complex-subset mean-patch Full Fusion configuration for ten seeds."""

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
from torch.utils.data import DataLoader

from rul_chronos.cache import EmbeddingCacheDataset
from rul_chronos.metrics import phm_score, rmse
from rul_chronos.model import WideDeepRULAdapter
from rul_chronos.training import evaluate_adapter, predict, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "FD002"
TRAIN_ENGINES = 208
VALIDATION_ENGINES = 52
TEST_ENGINES = 259
NUM_REGIMES = 6
SEED42_ELAPSED_SECONDS: float | None = None
CACHE_DIR = PROJECT_ROOT / "artifacts" / DATASET
CACHE_MANIFEST = CACHE_DIR / "manifest.json"
OUTPUT_DIR = CACHE_DIR
RESULTS_JSON = CACHE_DIR / "seed_results.json"
RESULTS_CSV = CACHE_DIR / "seed_results.csv"
REPORT_PATH = CACHE_DIR / "TEN_SEED_REPORT.md"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
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
METRICS = ("best_val_rmse", "val_endpoint_rmse", "test_rmse", "test_phm_score")


def checkpoint_path(seed: int) -> Path:
    return OUTPUT_DIR / f"adapter_wide_deep_seed{seed}.pt"


def prediction_path(seed: int) -> Path:
    return checkpoint_path(seed).with_suffix(".predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "std_sample": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def previous_rows() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if RESULTS_JSON.exists():
        document = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        rows = {
            int(row["seed"]): row
            for row in document.get("runs", [])
            if row.get("status") == "completed"
        }
    seed42_summary = OUTPUT_DIR / "run_summary_seed42.json"
    if seed42_summary.exists() and rows.get(42, {}).get("elapsed_seconds") is None:
        original = json.loads(seed42_summary.read_text(encoding="utf-8"))["run"]
        rows.setdefault(42, {})["elapsed_seconds"] = original.get("elapsed_seconds")
    if rows.get(42, {}).get("elapsed_seconds") is None and SEED42_ELAPSED_SECONDS is not None:
        rows.setdefault(42, {})["elapsed_seconds"] = SEED42_ELAPSED_SECONDS
    return rows


def validation_endpoint_rmse(saved: dict[str, Any]) -> float:
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
        raise RuntimeError("recomputed validation RMSE differs from the selected checkpoint")
    return rmse(target[endpoint_indices], prediction[endpoint_indices])


def verified_row(seed: int, elapsed_seconds: float | None) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(seed)
    predictions = prediction_path(seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_kwargs = {
        "num_sensors": 21,
        "embedding_dim": 768,
        "compression_dim": 8,
        "num_regimes": NUM_REGIMES,
        "dropout": 0.2,
        "deep_only": False,
    }
    if int(saved["seed"]) != seed or int(saved.get("min_prefix_length", 1)) != 1:
        raise RuntimeError(f"seed {seed} checkpoint metadata differs")
    if saved["model_kwargs"] != expected_kwargs:
        raise RuntimeError(f"seed {seed} model configuration differs")
    data = json.loads(predictions.read_text(encoding="utf-8"))
    units = np.asarray(data["unit"], dtype=np.int64)
    target = np.asarray(data["target"], dtype=np.float64)
    prediction = np.asarray(data["prediction"], dtype=np.float64)
    if (
        len(prediction) != TEST_ENGINES
        or len(np.unique(units)) != TEST_ENGINES
        or not np.isfinite(prediction).all()
    ):
        raise RuntimeError(
            f"seed {seed} does not contain {TEST_ENGINES} valid unique-engine predictions"
        )
    test_rmse = rmse(target, prediction)
    score = phm_score(target, prediction)
    stored = data["metrics"]
    if not np.isclose(test_rmse, stored["rmse"], rtol=0.0, atol=1e-10):
        raise RuntimeError(f"seed {seed} stored RMSE differs")
    if not np.isclose(score, stored["score"], rtol=0.0, atol=1e-10):
        raise RuntimeError(f"seed {seed} stored Score differs")
    return {
        "seed": seed,
        "status": "completed",
        "best_epoch": int(saved["best_epoch"]),
        "completed_epochs": int(saved["completed_epochs"]),
        "best_val_rmse": float(saved["best_val_rmse"]),
        "val_endpoint_rmse": float(validation_endpoint_rmse(saved)),
        "test_rmse": float(test_rmse),
        "test_phm_score": float(score),
        "prediction_count": int(len(prediction)),
        "unique_test_engines": int(len(np.unique(units))),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": relative(checkpoint),
        "predictions": relative(predictions),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "completed"]
    result: dict[str, Any] = {
        "count": len(complete),
        "epoch_limit": int(TRAINING["epochs"]),
        "runs_completed_epoch_limit": sum(
            int(row["completed_epochs"]) == int(TRAINING["epochs"]) for row in complete
        ),
    }
    if complete:
        result.update({metric: metric_summary([float(row[metric]) for row in complete]) for metric in METRICS})
        seed42 = next((row for row in complete if row["seed"] == 42), None)
        if seed42 is not None:
            result["seed42_test_rmse_rank_ascending"] = 1 + sum(
                row["test_rmse"] < seed42["test_rmse"] for row in complete
            )
            result["seed42_score_rank_ascending"] = 1 + sum(
                row["test_phm_score"] < seed42["test_phm_score"] for row in complete
            )
    return result


def save(rows_by_seed: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_seed[seed] for seed in SEEDS if seed in rows_by_seed]
    document = {
        "dataset": DATASET,
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion",
        "engine_split": relative(CACHE_DIR / "split.json"),
        "embedding_cache": relative(CACHE_DIR),
        "seeds": SEEDS,
        "fixed_training": TRAINING,
        "validation_selection_metric": "RMSE over all validation prefixes",
        "standard_deviation_note": "std_sample uses ddof=1; std_population uses ddof=0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    atomic_write(RESULTS_JSON, json.dumps(document, indent=2) + "\n")
    fields = [
        "seed", "status", "best_epoch", "completed_epochs", "best_val_rmse",
        "val_endpoint_rmse", "test_rmse", "test_phm_score", "prediction_count",
        "unique_test_engines", "elapsed_seconds", "checkpoint", "predictions",
    ]
    temporary = RESULTS_CSV.with_suffix(RESULTS_CSV.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(RESULTS_CSV)
    if len(rows) == len(SEEDS):
        summary = document["aggregate"]
        labels = {
            "best_val_rmse": "Validation all-prefix RMSE",
            "val_endpoint_rmse": "Validation endpoint RMSE",
            "test_rmse": "Test RMSE",
            "test_phm_score": "PHM Score",
        }
        lines = [
            f"# {DATASET} Valid-patch-mean Full Fusion：10 seeds",
            "",
            f"固定同一份{TRAIN_ENGINES}/{VALIDATION_ENGINES}发动机划分、同一 Chronos-2 embedding缓存、"
            "21-sensor联合 group attention、",
            "无限制 prefix、`dcomp=8`、AdamW、lr `1e-4`、weight decay `1e-3`、batch size 1024、",
            f"dropout 0.2、最多{TRAINING['epochs']} epochs、patience 15。"
            "均值后的 `±` 为样本标准差（ddof=1）。",
            "",
            "## 汇总",
            "",
            "| 指标 | Mean ± SD | Min | Max |",
            "| --- | ---: | ---: | ---: |",
        ]
        for metric in METRICS:
            item = summary[metric]
            lines.append(
                f"| {labels[metric]} | {item['mean']:.6f} ± {item['std_sample']:.6f} | "
                f"{item['min']:.6f} | {item['max']:.6f} |"
            )
        lines.extend([
            "",
            "验证轨迹完整运行至失效，endpoint 标签均为0；官方测试轨迹提前截断，",
            "因此 validation endpoint RMSE 不能与 test RMSE 直接比较，也不用于 early stopping。",
            "",
            "## 逐 seed",
            "",
            "| Seed | Best epoch | Completed | Val all-prefix | Val endpoint | Test RMSE | PHM Score |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in rows:
            lines.append(
                f"| {row['seed']} | {row['best_epoch']} | {row['completed_epochs']} | "
                f"{row['best_val_rmse']:.6f} | {row['val_endpoint_rmse']:.6f} | "
                f"{row['test_rmse']:.6f} | {row['test_phm_score']:.6f} |"
            )
        atomic_write(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> None:
    if not CACHE_MANIFEST.exists():
        raise RuntimeError(f"{DATASET} embedding cache is missing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = previous_rows()
    rows_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        prior_elapsed = previous.get(seed, {}).get("elapsed_seconds")
        row = verified_row(seed, prior_elapsed)
        if row is not None:
            rows_by_seed[seed] = row
    save(rows_by_seed)
    print(f"{DATASET} progress: {len(rows_by_seed)}/{len(SEEDS)} completed")

    for seed in SEEDS:
        if seed in rows_by_seed:
            continue
        checkpoint = checkpoint_path(seed)
        log_path = OUTPUT_DIR / f"adapter_wide_deep_seed{seed}.training.log"
        started = time.monotonic()
        print(f"starting seed={seed}; log={relative(log_path)}")
        with log_path.open("w", encoding="utf-8") as output, redirect_stdout(output):
            train_adapter(CACHE_DIR, checkpoint, seed=seed, **TRAINING)
            evaluate_adapter(CACHE_DIR, checkpoint, TRAINING["batch_size"], TRAINING["device_name"])
        elapsed = round(time.monotonic() - started, 3)
        row = verified_row(seed, elapsed)
        assert row is not None
        rows_by_seed[seed] = row
        save(rows_by_seed)
        print(
            f"completed seed={seed}: epoch={row['best_epoch']}/{row['completed_epochs']} "
            f"val={row['best_val_rmse']:.6f} test={row['test_rmse']:.6f} "
            f"score={row['test_phm_score']:.6f} seconds={elapsed:.1f}"
        )

    if set(rows_by_seed) != set(SEEDS):
        raise RuntimeError(f"{DATASET} ten-seed sweep is incomplete")
    save(rows_by_seed)
    print(json.dumps(aggregate(list(rows_by_seed.values())), indent=2))


if __name__ == "__main__":
    main()
