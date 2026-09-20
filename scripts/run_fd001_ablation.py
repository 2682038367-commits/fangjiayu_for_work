#!/usr/bin/env python3
"""Run the FD001 fusion and compression-dimension ablations."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rul_chronos.cache import EmbeddingCacheDataset
from rul_chronos.metrics import phm_score, rmse
from rul_chronos.model import WideDeepRULAdapter
from rul_chronos.training import _cache_on_device, _predict_cached, evaluate_adapter, train_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = "FD001"
TEST_ENGINES = 100
NUM_REGIMES = 1
CACHE_DIR = PROJECT_ROOT / "artifacts" / "FD001"
OUTPUT_DIR = CACHE_DIR / "ablations"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
COMPRESSION_DIMS = [2, 4, 8, 16, 128, 256]
CONFIGS = [*(('fusion', dim) for dim in COMPRESSION_DIMS), ('deep_only', 8)]
CSV_PATH = OUTPUT_DIR / "ablation_runs.csv"
JSON_PATH = OUTPUT_DIR / "ablation_summary.json"
REPORT_PATH = OUTPUT_DIR / "ABLATION_REPORT_GENERATED.md"

TRAINING = {
    "batch_size": 1024,
    "epochs": 150,
    "patience": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
    "dropout": 0.2,
    "device_name": "auto",
}


def checkpoint_path(mode: str, dim: int, seed: int) -> Path:
    if mode == "fusion" and dim == 8:
        return CACHE_DIR / f"adapter_wide_deep_seed{seed}.pt"
    return OUTPUT_DIR / f"adapter_{mode}_dim{dim}_seed{seed}.pt"


def prediction_path(mode: str, dim: int, seed: int) -> Path:
    return checkpoint_path(mode, dim, seed).with_suffix(".predictions.json")


def validation_prediction_path(mode: str, dim: int, seed: int) -> Path:
    return checkpoint_path(mode, dim, seed).with_suffix(".validation_predictions.json")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_existing_rows() -> dict[tuple[str, int, int], dict[str, Any]]:
    if not JSON_PATH.exists():
        return {}
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {
        (row["mode"], int(row["compression_dim"]), int(row["seed"])): row
        for row in document.get("runs", [])
        if row.get("status") == "completed"
    }


def load_completed(mode: str, dim: int, seed: int, previous: dict[str, Any] | None) -> dict[str, Any] | None:
    checkpoint = checkpoint_path(mode, dim, seed)
    predictions = prediction_path(mode, dim, seed)
    if not checkpoint.exists() or not predictions.exists():
        return None
    saved_model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_predictions = json.loads(predictions.read_text(encoding="utf-8"))
    expected_kwargs = {
        "num_sensors": 21,
        "embedding_dim": 768,
        "compression_dim": dim,
        "num_regimes": NUM_REGIMES,
        "dropout": 0.2,
        "deep_only": mode == "deep_only",
    }
    if int(saved_model["seed"]) != seed or saved_model["model_kwargs"] != expected_kwargs:
        raise RuntimeError(f"checkpoint configuration differs for {(mode, dim, seed)}")
    units = np.asarray(saved_predictions["unit"], dtype=np.int64)
    target = np.asarray(saved_predictions["target"], dtype=np.float64)
    prediction = np.asarray(saved_predictions["prediction"], dtype=np.float64)
    if len(prediction) != TEST_ENGINES or len(np.unique(units)) != TEST_ENGINES:
        raise RuntimeError(f"prediction coverage differs for {(mode, dim, seed)}")
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite prediction for {(mode, dim, seed)}")
    metrics = {"rmse": rmse(target, prediction), "score": phm_score(target, prediction)}
    stored_metrics = saved_predictions["metrics"]
    for key, value in metrics.items():
        if not np.isclose(value, stored_metrics[key], rtol=0.0, atol=1e-10):
            raise RuntimeError(f"stored {key} differs for {(mode, dim, seed)}")
    row = {
        "mode": mode,
        "compression_dim": dim,
        "seed": seed,
        "status": "completed",
        "best_val_rmse": float(saved_model["best_val_rmse"]),
        "test_rmse": float(metrics["rmse"]),
        "test_phm_score": float(metrics["score"]),
        "elapsed_seconds": None if previous is None else previous.get("elapsed_seconds"),
        "checkpoint": relative(checkpoint),
        "predictions": relative(predictions),
    }
    if previous is not None:
        for key in ("val_all_prefix_rmse", "val_endpoint_rmse", "validation_predictions"):
            if key in previous:
                row[key] = previous[key]
    return row


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for mode, dim in CONFIGS:
        selected = [
            row for row in rows
            if row["mode"] == mode and row["compression_dim"] == dim and row["status"] == "completed"
        ]
        if not selected:
            continue
        groups.append({
            "mode": mode,
            "compression_dim": dim,
            "count": len(selected),
            "best_val_rmse": metric_summary(selected, "best_val_rmse"),
            "test_rmse": metric_summary(selected, "test_rmse"),
            "test_phm_score": metric_summary(selected, "test_phm_score"),
        })
        for key in ("val_all_prefix_rmse", "val_endpoint_rmse"):
            metric_rows = [row for row in selected if key in row]
            if metric_rows:
                groups[-1][key] = metric_summary(metric_rows, key)
    return groups


def conclusions(rows: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(group["mode"], group["compression_dim"]): group for group in groups}
    dimension_groups = [lookup[("fusion", dim)] for dim in COMPRESSION_DIMS if ("fusion", dim) in lookup]
    best_by_all_prefix = min(dimension_groups, key=lambda group: group["best_val_rmse"]["mean"]) if dimension_groups else None
    endpoint_groups = [group for group in dimension_groups if "val_endpoint_rmse" in group]
    best_by_endpoint = min(endpoint_groups, key=lambda group: group["val_endpoint_rmse"]["mean"]) if endpoint_groups else None

    paired_deltas = []
    fusion_wins = 0
    fusion_val_wins = 0
    fusion_score_wins = 0
    for seed in SEEDS:
        fusion = next((row for row in rows if row["mode"] == "fusion" and row["compression_dim"] == 8 and row["seed"] == seed), None)
        deep = next((row for row in rows if row["mode"] == "deep_only" and row["compression_dim"] == 8 and row["seed"] == seed), None)
        if fusion is not None and deep is not None:
            delta = float(deep["test_rmse"] - fusion["test_rmse"])
            val_delta = float(deep["best_val_rmse"] - fusion["best_val_rmse"])
            score_delta = float(deep["test_phm_score"] - fusion["test_phm_score"])
            paired_deltas.append({
                "seed": seed,
                "deep_minus_fusion_val_all_prefix_rmse": val_delta,
                "deep_minus_fusion_test_rmse": delta,
                "deep_minus_fusion_test_phm_score": score_delta,
            })
            fusion_wins += int(delta > 0)
            fusion_val_wins += int(val_delta > 0)
            fusion_score_wins += int(score_delta > 0)

    delta_values = np.asarray([item["deep_minus_fusion_test_rmse"] for item in paired_deltas], dtype=np.float64)
    paired_count = len(paired_deltas)
    losses = paired_count - fusion_wins
    more_extreme_side = min(fusion_wins, losses)
    sign_test_p = (
        min(1.0, 2.0 * sum(math.comb(paired_count, k) for k in range(more_extreme_side + 1)) / (2**paired_count))
        if paired_count else None
    )
    fusion_group = lookup.get(("fusion", 8))
    deep_group = lookup.get(("deep_only", 8))
    fusion_mean = None if fusion_group is None else fusion_group["test_rmse"]["mean"]
    deep_mean = None if deep_group is None else deep_group["test_rmse"]["mean"]
    return {
        "best_fusion_dimension_by_mean_val_all_prefix_rmse": None if best_by_all_prefix is None else best_by_all_prefix["compression_dim"],
        "best_fusion_dimension_by_mean_val_endpoint_rmse": None if best_by_endpoint is None else best_by_endpoint["compression_dim"],
        "val_endpoint_selection_warning": (
            f"{DATASET} validation engines run to failure, so every final-cycle validation target is 0; "
            "official test endpoints are censored before failure (targets 7..125). The endpoint ranking "
            "is reported but is not a like-for-like model-selection criterion."
        ),
        "fusion_vs_deep_only_dim8": {
            "paired_count": len(paired_deltas),
            "fusion_wins_on_val_all_prefix_rmse": fusion_val_wins,
            "fusion_wins_on_test_rmse": fusion_wins,
            "fusion_wins_on_test_phm_score": fusion_score_wins,
            "mean_deep_minus_fusion_val_all_prefix_rmse": (
                float(np.mean([item["deep_minus_fusion_val_all_prefix_rmse"] for item in paired_deltas]))
                if paired_deltas else None
            ),
            "mean_deep_minus_fusion_test_rmse": float(delta_values.mean()) if len(delta_values) else None,
            "mean_deep_minus_fusion_test_phm_score": (
                float(np.mean([item["deep_minus_fusion_test_phm_score"] for item in paired_deltas]))
                if paired_deltas else None
            ),
            "fusion_test_rmse_mean": fusion_mean,
            "deep_only_test_rmse_mean": deep_mean,
            "fusion_relative_rmse_reduction_percent": (
                None if fusion_mean is None or deep_mean is None else float(100.0 * (deep_mean - fusion_mean) / deep_mean)
            ),
            "two_sided_exact_sign_test_p": sign_test_p,
            "paired_deltas": paired_deltas,
        },
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save(rows_by_key: dict[tuple[str, int, int], dict[str, Any]]) -> None:
    order = {(mode, dim): index for index, (mode, dim) in enumerate(CONFIGS)}
    rows = sorted(rows_by_key.values(), key=lambda row: (order[(row["mode"], row["compression_dim"])], row["seed"]))
    groups = summarize(rows)
    document = {
        "dataset": DATASET,
        "seeds": SEEDS,
        "compression_dimensions": COMPRESSION_DIMS,
        "design": {
            "compression_grid": "full fusion for every compression dimension",
            "fusion_ablation": "full fusion versus only deep at compression_dim=8",
        },
        "training": TRAINING,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "groups": groups,
        "conclusions": conclusions(rows, groups),
    }
    atomic_write(JSON_PATH, json.dumps(document, indent=2) + "\n")

    fields = [
        "mode", "compression_dim", "seed", "status", "best_val_rmse",
        "val_all_prefix_rmse", "val_endpoint_rmse", "test_rmse", "test_phm_score",
        "elapsed_seconds", "checkpoint", "predictions", "validation_predictions",
    ]
    temporary = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(CSV_PATH)
    if (
        len(rows) == len(CONFIGS) * len(SEEDS)
        and all(row.get("status") == "completed" and "val_endpoint_rmse" in row for row in rows)
    ):
        save_report(document)


def save_report(document: dict[str, Any]) -> None:
    groups = {(group["mode"], group["compression_dim"]): group for group in document["groups"]}
    conclusion = document["conclusions"]
    lines = [
        f"# {DATASET} Mean-patch 消融实验",
        "",
        "固定同一 embedding缓存、发动机划分和10个 seeds。Full Fusion比较压缩维度",
        "`2, 4, 8, 16, 128, 256`；并在`dcomp=8`比较Full Fusion与Only Deep。",
        "训练采用AdamW、lr `1e-4`、weight decay `1e-3`、batch size 1024、dropout 0.2、",
        "最多150 epochs、patience 15。均值后的 `±` 为样本标准差（ddof=1）。",
        "",
        "## 压缩维度：Full Fusion",
        "",
        "| dcomp | Val all-prefix RMSE | Val endpoint RMSE | Test RMSE | PHM Score |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for dim in COMPRESSION_DIMS:
        group = groups[("fusion", dim)]
        lines.append(
            f"| {dim} | {group['best_val_rmse']['mean']:.6f} ± {group['best_val_rmse']['std_sample']:.6f} | "
            f"{group['val_endpoint_rmse']['mean']:.6f} ± {group['val_endpoint_rmse']['std_sample']:.6f} | "
            f"{group['test_rmse']['mean']:.6f} ± {group['test_rmse']['std_sample']:.6f} | "
            f"{group['test_phm_score']['mean']:.6f} ± {group['test_phm_score']['std_sample']:.6f} |"
        )
    selected = conclusion["best_fusion_dimension_by_mean_val_all_prefix_rmse"]
    lines.extend([
        "",
        f"按预先指定的平均验证全前缀RMSE选择，最佳压缩维度为 `{selected}`。",
        "验证发动机完整运行至失效，endpoint标签均为0，因此endpoint排名不能与官方测试终点直接比较。",
        "",
        "## dcomp=8：Full Fusion vs Only Deep",
        "",
        "| Model | Val all-prefix RMSE | Test RMSE | PHM Score |",
        "| --- | ---: | ---: | ---: |",
    ])
    for mode, label in (("fusion", "Full Fusion"), ("deep_only", "Only Deep")):
        group = groups[(mode, 8)]
        lines.append(
            f"| {label} | {group['best_val_rmse']['mean']:.6f} ± {group['best_val_rmse']['std_sample']:.6f} | "
            f"{group['test_rmse']['mean']:.6f} ± {group['test_rmse']['std_sample']:.6f} | "
            f"{group['test_phm_score']['mean']:.6f} ± {group['test_phm_score']['std_sample']:.6f} |"
        )
    paired = conclusion["fusion_vs_deep_only_dim8"]
    lines.extend([
        "",
        f"Full Fusion在验证全前缀RMSE、Test RMSE、PHM Score上分别赢得 "
        f"`{paired['fusion_wins_on_val_all_prefix_rmse']}/10`、"
        f"`{paired['fusion_wins_on_test_rmse']}/10`、"
        f"`{paired['fusion_wins_on_test_phm_score']}/10` 个配对 seed。",
        f"平均 `Only Deep−Full Fusion` 验证RMSE为 "
        f"`{paired['mean_deep_minus_fusion_val_all_prefix_rmse']:.6f}`；",
        f"平均 `Only Deep−Full Fusion` RMSE为 `{paired['mean_deep_minus_fusion_test_rmse']:.6f}`，",
        f"平均 `Only Deep−Full Fusion` Score为 `{paired['mean_deep_minus_fusion_test_phm_score']:.6f}`，",
        f"相对改善 `{paired['fusion_relative_rmse_reduction_percent']:.3f}%`，",
        f"双侧精确符号检验 `p={paired['two_sided_exact_sign_test_p']:.6f}`。",
        "",
    ])
    atomic_write(REPORT_PATH, "\n".join(lines))


def add_validation_metrics(rows_by_key: dict[tuple[str, int, int], dict[str, Any]]) -> None:
    pending = [key for key, row in rows_by_key.items() if "val_endpoint_rmse" not in row]
    print(f"validation endpoint evaluation: {len(pending)} checkpoints pending")
    if not pending:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation = EmbeddingCacheDataset(CACHE_DIR, "val")
    units = np.asarray(validation.units)
    cycles = np.asarray(validation.cycles)
    endpoint_indices = np.asarray(
        [np.flatnonzero(units == unit)[np.argmax(cycles[units == unit])] for unit in np.unique(units)],
        dtype=np.int64,
    )
    if len(endpoint_indices) != len(np.unique(units)):
        raise RuntimeError("Expected exactly one validation endpoint per engine")

    tensors = _cache_on_device(validation, device)
    endpoint_index_tensor = torch.as_tensor(endpoint_indices, dtype=torch.int64, device=device)
    endpoint_tensors = {key: value[endpoint_index_tensor] for key, value in tensors.items()}
    for number, key in enumerate(pending, start=1):
        mode, dim, seed = key
        row = rows_by_key[key]
        checkpoint = torch.load(checkpoint_path(mode, dim, seed), map_location="cpu", weights_only=True)
        model = WideDeepRULAdapter(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        all_target, all_prediction = _predict_cached(model, tensors, TRAINING["batch_size"])
        endpoint_target, endpoint_prediction = _predict_cached(model, endpoint_tensors, TRAINING["batch_size"])
        all_prefix_rmse = rmse(all_target, all_prediction)
        endpoint_rmse = rmse(endpoint_target, endpoint_prediction)
        if not np.isclose(all_prefix_rmse, row["best_val_rmse"], rtol=0.0, atol=1e-5):
            raise RuntimeError(
                f"Recomputed all-prefix RMSE differs for {key}: {all_prefix_rmse} vs {row['best_val_rmse']}"
            )
        output_path = validation_prediction_path(mode, dim, seed)
        output_path.write_text(
            json.dumps({
                "metrics": {
                    "val_all_prefix_rmse": all_prefix_rmse,
                    "val_endpoint_rmse": endpoint_rmse,
                },
                "unit": units[endpoint_indices].astype(int).tolist(),
                "cycle": cycles[endpoint_indices].astype(int).tolist(),
                "target": endpoint_target.tolist(),
                "prediction": endpoint_prediction.tolist(),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        row["val_all_prefix_rmse"] = float(all_prefix_rmse)
        row["val_endpoint_rmse"] = float(endpoint_rmse)
        row["validation_predictions"] = relative(output_path)
        save(rows_by_key)
        print(
            f"validation {number}/{len(pending)}: mode={mode} dim={dim} seed={seed} "
            f"all={all_prefix_rmse:.4f} endpoint={endpoint_rmse:.4f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_existing_rows()
    rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for mode, dim in CONFIGS:
        for seed in SEEDS:
            key = (mode, dim, seed)
            completed = load_completed(mode, dim, seed, previous.get(key))
            if completed is not None:
                rows_by_key[key] = completed
    save(rows_by_key)

    total = len(CONFIGS) * len(SEEDS)
    print(f"ablation progress: {len(rows_by_key)}/{total} runs already completed")
    for mode, dim in CONFIGS:
        for seed in SEEDS:
            key = (mode, dim, seed)
            if key in rows_by_key:
                continue
            checkpoint = checkpoint_path(mode, dim, seed)
            log_path = OUTPUT_DIR / f"adapter_{mode}_dim{dim}_seed{seed}.training.log"
            started = time.monotonic()
            print(f"starting mode={mode} dim={dim} seed={seed}")
            try:
                with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                    train_result = train_adapter(
                        CACHE_DIR,
                        checkpoint,
                        seed=seed,
                        compression_dim=dim,
                        deep_only=(mode == "deep_only"),
                        **TRAINING,
                    )
                    test_result = evaluate_adapter(
                        CACHE_DIR,
                        checkpoint,
                        batch_size=TRAINING["batch_size"],
                        device_name=TRAINING["device_name"],
                    )
                elapsed = time.monotonic() - started
                rows_by_key[key] = {
                    "mode": mode,
                    "compression_dim": dim,
                    "seed": seed,
                    "status": "completed",
                    "best_val_rmse": float(train_result["best_val_rmse"]),
                    "test_rmse": float(test_result["rmse"]),
                    "test_phm_score": float(test_result["score"]),
                    "elapsed_seconds": round(elapsed, 3),
                    "checkpoint": relative(checkpoint),
                    "predictions": relative(prediction_path(mode, dim, seed)),
                }
                save(rows_by_key)
                print(
                    f"completed {len(rows_by_key)}/{total}: mode={mode} dim={dim} seed={seed} "
                    f"val={train_result['best_val_rmse']:.4f} test={test_result['rmse']:.4f} "
                    f"score={test_result['score']:.2f} seconds={elapsed:.1f}"
                )
            except Exception as error:
                rows_by_key[key] = {
                    "mode": mode,
                    "compression_dim": dim,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "checkpoint": relative(checkpoint),
                    "predictions": relative(prediction_path(mode, dim, seed)),
                }
                save(rows_by_key)
                raise

    save(rows_by_key)
    add_validation_metrics(rows_by_key)
    save(rows_by_key)
    print(json.dumps(json.loads(JSON_PATH.read_text(encoding="utf-8"))["conclusions"], indent=2))


if __name__ == "__main__":
    main()
