#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report only identifiable comparisons for the MetaIndux-TS reproduction.

The paper reports a predictive score obtained by training on generated data,
but does not report its real-data training baseline.  Therefore a protocol
difference (paper-real minus local-real) and the paper's synthetic-data
penalty are unidentifiable.  This script deliberately does not perform a
cross-paper t test or a “protocol-layer” decomposition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
PAPER_METRICS_MANIFEST = PROJECT / "docs" / "provenance" / "paper_predictive_scores_w48.json"
SYNTHETIC_SUMMARY = {
    "FD002": PROJECT / "results" / "frequency_threshold_validation"
    / "fd002_public_arm_seven_seeds_evaluation_summary.csv",
    "FD003": PROJECT / "results" / "fd003_w48_public_code_evaluation_summary.csv",
    "FD004": PROJECT / "results" / "fd004_w48_public_code_evaluation_summary.csv",
}
REAL_BASELINE = PROJECT / "results" / "real_data_baseline_w48.csv"


def read_paper_generated_rmse(path: Path):
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)["values"]
    result = {str(dataset).upper(): float(value) for dataset, value in values.items()}
    if set(result) != {"FD002", "FD003", "FD004"}:
        raise ValueError(f"unexpected paper metric datasets: {sorted(result)}")
    return result


def read_synthetic_means(path: Path):
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "rmse" and row["generation_seed"] != "aggregate":
                values.append(float(row["mean"]))
    if not values:
        raise ValueError(f"missing per-generation RMSE rows: {path}")
    return np.asarray(values)


def read_real_baselines(path: Path, datasets):
    data = {dataset: [] for dataset in datasets}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in data:
                data[row["dataset"]].append(float(row["rmse"]))
    if any(not values for values in data.values()):
        raise ValueError(f"missing real baseline rows: {path}")
    return {dataset: np.asarray(values) for dataset, values in data.items()}


def main():
    paper_generated_rmse = read_paper_generated_rmse(PAPER_METRICS_MANIFEST)
    real = read_real_baselines(REAL_BASELINE, paper_generated_rmse)
    print("可识别比较：论文生成结果、本地生成结果与本地真实数据基线")
    print("=" * 112)
    print(f"{'数据集':<8}{'论文生成点估计':>15}{'本地生成均值':>14}{'本地真实均值':>14}"
          f"{'正式复现差距':>14}{'本地生成数据惩罚':>18}")
    print("-" * 112)
    for dataset in ("FD002", "FD003", "FD004"):
        synthetic = read_synthetic_means(SYNTHETIC_SUMMARY[dataset])
        local_synthetic = float(synthetic.mean())
        local_real = float(real[dataset].mean())
        reproduction_gap = local_synthetic - paper_generated_rmse[dataset]
        local_penalty = local_synthetic - local_real
        print(f"{dataset:<8}{paper_generated_rmse[dataset]:>15.3f}{local_synthetic:>14.3f}"
              f"{local_real:>14.3f}{reproduction_gap:>+14.3f}{local_penalty:>+18.3f}")
    print("-" * 112)
    print("正式复现差距 = 本地生成数据训练 RMSE − 论文生成数据训练 RMSE。")
    print("本地生成数据惩罚 = 本地生成数据训练 RMSE − 本地真实数据训练 RMSE。")
    print()
    print("不可识别的量（论文未报告真实数据训练基线）：")
    print("  - 论文生成数据相对论文真实数据的性能损失")
    print("  - 论文真实数据训练与本地真实数据训练之间的协议差异")
    print("因此，不计算“协议层”、生成端占比、跨论文 p 值、Q/I² 或 FD002/FD004 协议差异。")
    print(f"论文点估计来源：{PAPER_METRICS_MANIFEST.relative_to(PROJECT)}（Table I，p. 18068）。")


if __name__ == "__main__":
    main()
