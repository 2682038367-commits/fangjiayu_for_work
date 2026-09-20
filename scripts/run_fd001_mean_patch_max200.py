#!/usr/bin/env python3
"""Run FD001 mean-patch Full Fusion for ten seeds with a 200-epoch limit."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np

import run_fd002_seed_sweep as sweep


sweep.DATASET = "FD001"
sweep.TRAIN_ENGINES = 80
sweep.VALIDATION_ENGINES = 20
sweep.TEST_ENGINES = 100
sweep.NUM_REGIMES = 1
sweep.SEED42_ELAPSED_SECONDS = None
sweep.CACHE_DIR = (
    sweep.PROJECT_ROOT / "artifacts" / "FD001" / "representation_ablation" / "valid_patch_mean"
)
sweep.CACHE_MANIFEST = sweep.CACHE_DIR.parent / "manifest.json"
sweep.OUTPUT_DIR = sweep.CACHE_DIR / "max200"
sweep.RESULTS_JSON = sweep.OUTPUT_DIR / "seed_results.json"
sweep.RESULTS_CSV = sweep.OUTPUT_DIR / "seed_results.csv"
sweep.REPORT_PATH = sweep.OUTPUT_DIR / "TEN_SEED_REPORT.md"
sweep.TRAINING = {**sweep.TRAINING, "epochs": 200}


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def save_paired_comparison() -> None:
    max150_path = sweep.CACHE_DIR / "seed_results.json"
    max200_path = sweep.RESULTS_JSON
    max150 = {row["seed"]: row for row in json.loads(max150_path.read_text())["runs"]}
    max200 = {row["seed"]: row for row in json.loads(max200_path.read_text())["runs"]}
    metrics = ("best_val_rmse", "test_rmse", "test_phm_score")
    rows = []
    for seed in sweep.SEEDS:
        row: dict[str, float | int] = {
            "seed": seed,
            "max150_best_epoch": int(max150[seed]["best_epoch"]),
            "max200_best_epoch": int(max200[seed]["best_epoch"]),
        }
        for metric in metrics:
            row[f"max150_{metric}"] = float(max150[seed][metric])
            row[f"max200_{metric}"] = float(max200[seed][metric])
            row[f"max150_minus_max200_{metric}"] = float(max150[seed][metric]) - float(max200[seed][metric])
        rows.append(row)
    comparison = {}
    for metric in metrics:
        before = [float(row[f"max150_{metric}"]) for row in rows]
        after = [float(row[f"max200_{metric}"]) for row in rows]
        delta = [float(row[f"max150_minus_max200_{metric}"]) for row in rows]
        comparison[metric] = {
            "max150": summarize(before),
            "max200": summarize(after),
            "paired_max150_minus_max200": summarize(delta),
            "max200_wins": sum(value > 0 for value in delta),
            "max150_wins": sum(value < 0 for value in delta),
            "ties": sum(value == 0 for value in delta),
        }
    document = {
        "dataset": "FD001",
        "representation": "valid_patch_mean",
        "configuration": "dcomp=8 Full Fusion",
        "design": "paired by seed; positive max150_minus_max200 values favor max200",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "paired_runs": rows,
        "comparison": comparison,
    }
    json_path = sweep.OUTPUT_DIR / "paired_max150_vs_max200.json"
    csv_path = sweep.OUTPUT_DIR / "paired_max150_vs_max200.csv"
    report_path = sweep.OUTPUT_DIR / "MAX150_VS_MAX200_REPORT.md"
    sweep.atomic_write(json_path, json.dumps(document, indent=2) + "\n")
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    labels = {
        "best_val_rmse": "Validation all-prefix RMSE",
        "test_rmse": "Test RMSE",
        "test_phm_score": "PHM Score",
    }
    lines = [
        "# FD001 Mean-patch：max150 vs max200",
        "",
        "除训练轮次上限外配置完全一致；逐 seed 配对。正的 `max150−max200` 表示200轮更好。",
        "均值后的 `±` 为样本标准差（ddof=1）。",
        "",
        "| 指标 | Max150 | Max200 | 配对差值（150−200） | Max200获胜 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        item = comparison[metric]
        lines.append(
            f"| {labels[metric]} | {item['max150']['mean']:.6f} ± {item['max150']['std_sample']:.6f} | "
            f"{item['max200']['mean']:.6f} ± {item['max200']['std_sample']:.6f} | "
            f"{item['paired_max150_minus_max200']['mean']:.6f} ± "
            f"{item['paired_max150_minus_max200']['std_sample']:.6f} | {item['max200_wins']}/10 |"
        )
    lines.extend([
        "",
        "| Seed | Epoch150 | Epoch200 | Test150 | Test200 | Score150 | Score200 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['max150_best_epoch']} | {row['max200_best_epoch']} | "
            f"{row['max150_test_rmse']:.6f} | {row['max200_test_rmse']:.6f} | "
            f"{row['max150_test_phm_score']:.6f} | {row['max200_test_phm_score']:.6f} |"
        )
    sweep.atomic_write(report_path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    sweep.main()
    save_paired_comparison()
