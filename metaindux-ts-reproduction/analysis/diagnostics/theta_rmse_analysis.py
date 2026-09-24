#!/usr/bin/env python3
"""Public-code frequency-quantile parameter versus RMSE diagnostic.

Scope is deliberately narrow: only ``hard_random_quantile`` checkpoints and
the formal five-evaluator result tables are used.  The two values in a
checkpoint are quantile parameters, not measured frequency-drop percentages.
Soft masks, STE masks, fixed-threshold arms, legacy built-in metrics, and
single-evaluator results are intentionally excluded.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch


PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "results" / "diagnostics"


@dataclass(frozen=True)
class Source:
    dataset: str
    summary_csv: Path


# Every source below is a generation-seed × five-evaluator formal summary.
SOURCES = (
    Source("FD001", PROJECT / "results"
           / "fd001_w48_public_code_evaluation_summary.csv"),
    Source("FD002", PROJECT / "results" / "frequency_threshold_validation"
           / "fd002_public_arm_seven_seeds_evaluation_summary.csv"),
    Source("FD003", PROJECT / "results"
           / "fd003_w48_public_code_evaluation_summary.csv"),
    Source("FD004", PROJECT / "results"
           / "fd004_w48_public_code_evaluation_summary.csv"),
)
FIXED_VS_RANDOM_PAIRED_CSV = (
    PROJECT / "results" / "frequency_threshold_validation" / "fd002_seven_seeds_paired.csv"
)


def exact_spearman(x, y):
    """Spearman rho and exact two-sided permutation p-value for n <= 8."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(stats.spearmanr(x, y).statistic)
    if len(x) > 8:
        return rho, float(stats.spearmanr(x, y).pvalue)
    exceedances = 0
    total = 0
    for order in itertools.permutations(range(len(y))):
        permuted = stats.spearmanr(x, y[list(order)]).statistic
        exceedances += abs(permuted) >= abs(rho) - 1e-12
        total += 1
    return rho, exceedances / total


def read_formal_rmse(path: Path):
    """Read only per-generation RMSE means and evaluator SDs from one summary."""
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != "rmse" or row["generation_seed"] == "aggregate":
                continue
            seed = int(row["generation_seed"])
            rows[seed] = {
                "rmse": float(row["mean"]),
                "evaluator_sd": float(row["sample_std"]),
            }
    if not rows:
        raise ValueError(f"no per-seed RMSE rows in {path}")
    return rows


def read_quantile_parameters(checkpoint: Path):
    """Read exactly the two public-code spectral quantile parameters."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    values = [
        (name, float(value.detach().cpu().item()))
        for name, value in state.items()
        if name.endswith("threshold_param")
    ]
    values.sort()
    if len(values) != 2:
        raise ValueError(f"expected two threshold parameters in {checkpoint}, got {values}")
    for _, value in values:
        if not 0.0 < value < 0.5:
            raise ValueError(
                f"{checkpoint} is not a public hard-quantile checkpoint: θ={value}")
    return values[0][1], values[1][1]


def collect_rows():
    rows = []
    for source in SOURCES:
        rmse_by_seed = read_formal_rmse(source.summary_csv)
        for seed, metrics in sorted(rmse_by_seed.items()):
            checkpoint = PROJECT / "checkpoints" / f"{source.dataset}_w48_seed{seed}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"formal result has no checkpoint: {checkpoint}")
            theta_1, theta_2 = read_quantile_parameters(checkpoint)
            rows.append({
                "dataset": source.dataset,
                "generation_seed": seed,
                "theta_1_quantile": theta_1,
                "theta_2_quantile": theta_2,
                "theta_mean_quantile": (theta_1 + theta_2) / 2,
                "rmse_five_evaluator_mean": metrics["rmse"],
                "rmse_five_evaluator_sd": metrics["evaluator_sd"],
                "rmse_source": str(source.summary_csv.relative_to(PROJECT)),
                "checkpoint": str(checkpoint.relative_to(PROJECT)),
            })
    return rows


def read_seven_seed_fixed_ablation(path: Path):
    """Aggregate the seven-seed matched fixed-vs-random evaluation table."""
    per_seed = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["generation_seed"])
            per_seed.setdefault(seed, {"fixed": [], "random": []})
            per_seed[seed]["fixed"].append(float(row["rmse_fixed"]))
            per_seed[seed]["random"].append(float(row["rmse_random"]))
    if sorted(per_seed) != [3, 13, 23, 33, 43, 53, 63]:
        raise ValueError(f"unexpected paired generation seeds in {path}: {sorted(per_seed)}")
    if any(len(values["fixed"]) != 5 or len(values["random"]) != 5
           for values in per_seed.values()):
        raise ValueError(f"{path} must contain five evaluator rows per generation seed")
    return [
        (seed, float(np.mean(values["fixed"])), float(np.mean(values["random"])))
        for seed, values in sorted(per_seed.items())
    ]


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(rows):
    lines = [
        "# Public-code quantile parameter vs RMSE",
        "",
        "Scope: `hard_random_quantile` public-code checkpoints only. Every RMSE "
        "is the mean of the same five evaluator seeds from the formal summary "
        "CSV for that dataset.",
        "",
        "`theta_1_quantile` and `theta_2_quantile` are quantile parameters. "
        "They are not measured frequency-drop ratios. This diagnostic is "
        "descriptive; it does not identify a causal effect of theta because "
        "other model initializations differ by generation seed.",
        "",
        "| Dataset | n | rho(theta1, RMSE) | p | rho(theta2, RMSE) | p | rho(mean theta, RMSE) | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        y = [row["rmse_five_evaluator_mean"] for row in group]
        r1, p1 = exact_spearman([row["theta_1_quantile"] for row in group], y)
        r2, p2 = exact_spearman([row["theta_2_quantile"] for row in group], y)
        rm, pm = exact_spearman([row["theta_mean_quantile"] for row in group], y)
        summary[dataset] = (r1, p1, r2, p2, rm, pm)
        lines.append(
            f"| {dataset} | {len(group)} | {r1:+.3f} | {p1:.3f} | "
            f"{r2:+.3f} | {p2:.3f} | {rm:+.3f} | {pm:.3f} |")
    lines.extend([
        "",
        "FD001 is now included because its completed table uses the same formal "
        "five-evaluator protocol. Fixed, soft, and STE arms are excluded by "
        "design rather than treated as cross-dataset correlation observations.",
    ])
    paired = read_seven_seed_fixed_ablation(FIXED_VS_RANDOM_PAIRED_CSV)
    fixed = np.asarray([item[1] for item in paired])
    random = np.asarray([item[2] for item in paired])
    delta = fixed - random
    test = stats.ttest_rel(fixed, random)
    wilcoxon = stats.wilcoxon(delta, alternative="two-sided", method="exact")
    lines.extend([
        "",
        "## FD002 fixed theta=0.25 paired ablation",
        "",
        "This is kept separate from the correlations above. It uses all seven "
        "matched generation seeds and five common evaluator seeds per generator.",
        "",
        "| Generation seed | fixed theta RMSE | random-quantile RMSE | fixed - random |",
        "|---:|---:|---:|---:|",
    ])
    for seed, fixed_rmse, random_rmse in paired:
        lines.append(f"| {seed} | {fixed_rmse:.6f} | {random_rmse:.6f} | "
                     f"{fixed_rmse-random_rmse:+.6f} |")
    lines.extend([
        "",
        f"Mean paired difference (fixed - random): {delta.mean():+.4f} RMSE; "
        f"sample SD: {delta.std(ddof=1):.4f}; paired t({len(delta)-1}) = "
        f"{test.statistic:+.3f}, p = {test.pvalue:.4f}; exact Wilcoxon p = "
        f"{wilcoxon.pvalue:.4f}.",
        "",
        "This ablation estimates the consequence of replacing random quantile "
        "parameters with 0.25 in this implementation. It does not establish "
        "that theta alone causes an individual seed's RMSE, nor does it alter "
        "the frozen public-code reproduction result.",
    ])
    return "\n".join(lines) + "\n", summary


def make_plot(rows, summary, path):
    datasets = sorted(summary)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.5 * len(datasets)),
                             squeeze=False)
    for row_index, dataset in enumerate(datasets):
        group = [row for row in rows if row["dataset"] == dataset]
        y = np.asarray([row["rmse_five_evaluator_mean"] for row in group])
        yerr = np.asarray([row["rmse_five_evaluator_sd"] for row in group])
        for column, (field, label, rho, pvalue) in enumerate((
            ("theta_1_quantile", "theta_1 quantile", summary[dataset][0], summary[dataset][1]),
            ("theta_2_quantile", "theta_2 quantile", summary[dataset][2], summary[dataset][3]),
        )):
            ax = axes[row_index, column]
            x = np.asarray([row[field] for row in group])
            ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
            for item, xv, yv in zip(group, x, y):
                ax.annotate(f"s{item['generation_seed']}", (xv, yv),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_title(f"{dataset}: rho={rho:+.3f}, exact p={pvalue:.3f}")
            ax.set_xlabel(label + " (not drop ratio)")
            ax.set_ylabel("formal five-evaluator RMSE mean")
            ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    rows = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "public_code_quantile_rmse.csv"
    report_path = OUT_DIR / "public_code_quantile_rmse_report.md"
    figure_path = OUT_DIR / "public_code_quantile_rmse.png"
    write_csv(rows, csv_path)
    report, summary = make_report(rows)
    report_path.write_text(report, encoding="utf-8")
    make_plot(rows, summary, figure_path)
    print(f"wrote {csv_path}")
    print(f"wrote {report_path}")
    print(f"wrote {figure_path}")
    print(report)


if __name__ == "__main__":
    main()
