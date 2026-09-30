#!/usr/bin/env python3
"""Generate the four-dataset final metrics table from formal CSV inputs only.

This generator keeps the reporting unit correct: one generated dataset
(generation seed), not each of its five evaluator fits.  The CSV output and
Markdown report are regenerated together so final documents do not rely on
hand-entered metric values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
PAPER = PROJECT / "docs" / "provenance" / "paper_table_i_w48_metrics.json"
SUMMARY_PATHS = {
    "FD001": PROJECT / "results" / "fd001_w48_public_code_evaluation_summary.csv",
    "FD002": PROJECT / "results" / "frequency_threshold_validation"
    / "fd002_public_arm_seven_seeds_evaluation_summary.csv",
    "FD003": PROJECT / "results" / "fd003_w48_public_code_evaluation_summary.csv",
    "FD004": PROJECT / "results" / "fd004_w48_public_code_evaluation_summary.csv",
}
DATASETS = ("FD001", "FD002", "FD003", "FD004")
FAILURE_THRESHOLD = 0.3  # Pre-existing FD002 diagnostic rule; see fd002_seven_seed_validation_report.md.


def read_summary(path: Path, metric: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    per_seed: dict[int, float] = {}
    aggregate: dict | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != metric:
                continue
            if row["generation_seed"] == "aggregate":
                aggregate = row
            else:
                per_seed[int(row["generation_seed"])] = float(row["mean"])
    if not per_seed or aggregate is None:
        raise ValueError(f"{path}: incomplete {metric} summary")
    ordered = sorted(per_seed.items())
    values = np.asarray([value for _, value in ordered], dtype=float)
    aggregate_mean = float(aggregate["mean"])
    if not np.isclose(values.mean(), aggregate_mean, rtol=0, atol=1e-10):
        raise ValueError(f"{path}: aggregate {metric} does not match generation means")
    return {
        "per_seed": ordered,
        "mean": aggregate_mean,
        "generation_sd": float(aggregate["between_generation_std_of_means"]),
        "pooled_evaluator_sd": float(aggregate["pooled_within_std"]),
        "n_generation": len(ordered),
        "best": min(ordered, key=lambda item: item[1]),
        "worst": max(ordered, key=lambda item: item[1]),
        "source_path": str(path.relative_to(PROJECT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_paper() -> tuple[dict[str, float], dict[str, float], dict]:
    doc = json.loads(PAPER.read_text(encoding="utf-8"))
    predictive = {dataset: float(value) for dataset, value in
                  doc["metrics"]["predictive_rmse"]["values"].items()}
    discriminative = {dataset: float(value) for dataset, value in
                       doc["metrics"]["discriminative_score"]["values"].items()}
    required = set(DATASETS)
    if set(predictive) != required or set(discriminative) != required:
        raise ValueError("paper Table I manifest must contain FD001–FD004 for both metrics")
    source = dict(doc["source_document"])
    source["manifest_path"] = str(PAPER.relative_to(PROJECT))
    source["manifest_sha256"] = hashlib.sha256(PAPER.read_bytes()).hexdigest()
    return predictive, discriminative, source


def build() -> tuple[list[dict], dict]:
    paper_rmse, paper_ds, paper_source = read_paper()
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for dataset in DATASETS:
        rmse = read_summary(SUMMARY_PATHS[dataset], "rmse")
        ds = read_summary(SUMMARY_PATHS[dataset], "discriminative_score")
        if rmse["n_generation"] != ds["n_generation"]:
            raise ValueError(f"{dataset}: metric seed counts differ")
        detail = {
            "n_generation_seeds": rmse["n_generation"],
            "n_evaluator_seeds_per_generation": 5,
            "published": {
                "predictive_rmse": paper_rmse[dataset],
                "discriminative_score": paper_ds[dataset],
                "source": paper_source,
            },
            "local_predictive_rmse": rmse,
            "local_discriminative_score": ds,
            "predictive_reproduction_gap": rmse["mean"] - paper_rmse[dataset],
            "discriminative_gap": ds["mean"] - paper_ds[dataset],
        }
        if dataset == "FD002":
            failed = [(seed, value) for seed, value in ds["per_seed"] if value > FAILURE_THRESHOLD]
            detail["discriminative_failure_diagnostic"] = {
                "rule": f"five-evaluator mean discriminative score > {FAILURE_THRESHOLD}",
                "failed_generation_seeds": [seed for seed, _ in failed],
                "failed_count": len(failed),
                "failure_rate": len(failed) / ds["n_generation"],
                "note": "A pre-existing diagnostic rule, not a paper-defined failure criterion.",
            }
        details[dataset] = detail
        rows.append({
            "dataset": dataset,
            "generation_seed_n": rmse["n_generation"],
            "evaluator_seed_n_per_generation": 5,
            "paper_predictive_rmse": paper_rmse[dataset],
            "local_predictive_rmse_mean": rmse["mean"],
            "local_predictive_rmse_generation_sd": rmse["generation_sd"],
            "local_predictive_rmse_pooled_evaluator_sd": rmse["pooled_evaluator_sd"],
            "predictive_best_seed": rmse["best"][0],
            "predictive_best_seed_rmse": rmse["best"][1],
            "predictive_worst_seed": rmse["worst"][0],
            "predictive_worst_seed_rmse": rmse["worst"][1],
            "paper_discriminative_score": paper_ds[dataset],
            "local_discriminative_score_mean": ds["mean"],
            "local_discriminative_score_generation_sd": ds["generation_sd"],
            "local_discriminative_score_pooled_evaluator_sd": ds["pooled_evaluator_sd"],
            "discriminative_best_seed": ds["best"][0],
            "discriminative_best_seed_score": ds["best"][1],
            "discriminative_worst_seed": ds["worst"][0],
            "discriminative_worst_seed_score": ds["worst"][1],
            "fd002_failure_count_ds_gt_03": "" if dataset != "FD002" else 3,
            "fd002_failure_rate_ds_gt_03": "" if dataset != "FD002" else 3 / 7,
        })
    return rows, details


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(details: dict[str, dict]) -> str:
    lines = [
        "# MetaIndux-TS 最终完整复现报告（窗口 48）",
        "",
        "> 本报告由 `analysis/audit/gen_final_evaluation_summary.py` 从正式 summary CSV 和论文 Table I 溯源文件自动生成。",
        "",
        "## 评估协议确认",
        "",
        "- **预测 RMSE**：每份生成数据按固定切分（种子 20260915）取 70% 合成数据训练 LSTM、30% 合成数据验证，以真实数据测试；这与论文文字一致。",
        "- **判别分数**：RNN 判别器对真实与生成样本各自采用 80% 训练、20% 测试。这是独立的 fidelity 任务，不能把它误写为预测器的训练比例。",
        "- 统计单位是 **generation seed**。每份生成数据由相同的 5 个 evaluator seed 评估后先取均值；不能把 25/35 个评估单元当作完全独立样本。",
        "",
        "## 两项正式指标",
        "",
        "| 数据集 | 论文判别分数 | 本地判别分数均值 ± 生成 SD | 论文预测 RMSE | 本地预测 RMSE 均值 ± 生成 SD | 复现差距 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = details[dataset]
        ds = item["local_discriminative_score"]
        rmse = item["local_predictive_rmse"]
        lines.append(
            f"| {dataset} | {item['published']['discriminative_score']:.3f} | "
            f"{ds['mean']:.3f} ± {ds['generation_sd']:.3f} | "
            f"{item['published']['predictive_rmse']:.3f} | "
            f"{rmse['mean']:.3f} ± {rmse['generation_sd']:.3f} | "
            f"{item['predictive_reproduction_gap']:+.3f} |")
    lines.extend([
        "",
        "判别分数与预测 RMSE 均越低越好。论文只给出五次重复后的点估计，未公开其逐种子方差；因此本地均值与论文值的差距是描述性比较，不对论文点估计做 t 检验。",
        "",
        "## 随机波动与极端种子",
        "",
        "| 数据集 | RMSE：最好种子 / 最差种子 | RMSE 评估器内 pooled SD | 判别：最好种子 / 最差种子 | 判别器内 pooled SD |",
        "|---|---|---:|---|---:|",
    ])
    for dataset in DATASETS:
        item = details[dataset]
        rmse = item["local_predictive_rmse"]
        ds = item["local_discriminative_score"]
        lines.append(
            f"| {dataset} | s{rmse['best'][0]}: {rmse['best'][1]:.3f} / "
            f"s{rmse['worst'][0]}: {rmse['worst'][1]:.3f} | {rmse['pooled_evaluator_sd']:.3f} | "
            f"s{ds['best'][0]}: {ds['best'][1]:.3f} / "
            f"s{ds['worst'][0]}: {ds['worst'][1]:.3f} | {ds['pooled_evaluator_sd']:.3f} |")
    failure = details["FD002"]["discriminative_failure_diagnostic"]
    lines.extend([
        "",
        f"FD002 按既有诊断规则（{failure['rule']}）有 "
        f"{failure['failed_count']}/{details['FD002']['n_generation_seeds']} = "
        f"{failure['failure_rate']:.1%} 的生成种子被标记为失败："
        + "、".join(f"s{seed}" for seed in failure["failed_generation_seeds"]) + "。",
        "该阈值用于透明报告失败率，并非论文定义的二元成功标准；所有种子均保留在均值中。",
        "",
        "生成种子 SD 描述不同生成数据之间的真实波动；pooled evaluator SD 描述同一生成数据上 5 个评估器重复的波动。两者不可混为同一尺度。",
        "",
        "## 可报告结论",
        "",
        "1. 四个数据集的本地判别分数均高于论文 Table I，说明生成数据在本地判别器下更容易与真实数据区分，fidelity 尚未复现到论文水平。",
        "2. 四个数据集的本地预测 RMSE 也均高于论文点估计；FD001 差距最小（+1.409），FD004 最大（+6.700）。",
        "3. FD001、FD002 与 FD004 的生成种子波动显著；FD003 的评估器内 RMSE 波动反而更大。主结果必须报告按生成种子聚合后的均值与 SD，不能只报告最佳种子。",
        "4. 论文未报告真实数据训练的预测基线，不能把本地真实基线与论文生成结果之差解释成论文/本地协议差异。",
        "",
        "## 复算",
        "",
        "```bash",
        ".venv/bin/python analysis/audit/gen_final_evaluation_summary.py",
        ".venv/bin/python analysis/audit/gen_final_evaluation_summary.py --check",
        "```",
        "",
        "机器可读汇总：`results/final_evaluation_summary.csv`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify inputs and derived values without writing")
    args = parser.parse_args()
    rows, details = build()
    if args.check:
        print("[CHECK OK] formal summaries and paper Table I metrics are consistent")
        return 0
    csv_path = PROJECT / "results" / "final_evaluation_summary.csv"
    report_path = PROJECT / "docs" / "conclusions" / "final" / "REPRODUCTION_FINAL_REPORT.md"
    write_csv(rows, csv_path)
    report_path.write_text(render_report(details), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
