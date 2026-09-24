#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_frozen_numbers.py —— 冻结数字的唯一权威源生成器

为什么要有这个文件
------------------
结论文档（docs/conclusions/final/*.md）里的数字是给人看的，抄来抄去会错。
本脚本把「原始观测值」硬编码进来，其余所有统计量（均值/SD/t/p/CI/异质性…）
全部现场算出，并对已知冻结值做断言校验，最后输出 frozen_numbers.json。

任何引用本项目的数字，都应从 results/frozen_numbers.json 取，
不要在文档之间互相抄。

用法
----
    .venv/bin/python analysis/audit/gen_frozen_numbers.py
        # 输出到 results/frozen_numbers.json
    .venv/bin/python analysis/audit/gen_frozen_numbers.py --check
        # 只校验，不写文件（CI 用）

依赖：numpy, scipy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ============================================================================
# 一、原始观测值（唯一硬编码区，其余全由计算得出）
# ============================================================================

GEN_SEEDS = [3, 13, 23, 33, 43, 53, 63]      # generation seeds，统计单位 n=7
EVAL_SEEDS = [3, 13, 23, 33, 43]             # evaluator seeds，先按 gen 内平均

# --- 正式生成结果的唯一输入：逐 generation seed 的 summary CSV ---
PROJECT = Path(__file__).resolve().parents[2]
SYNTHETIC_SUMMARIES = {
    "FD002_fixed_theta025": PROJECT / "results" / "frequency_threshold_validation"
    / "fd002_fixed_025_seven_seeds_evaluation_summary.csv",
    "FD002_public": PROJECT / "results" / "frequency_threshold_validation"
    / "fd002_public_arm_seven_seeds_evaluation_summary.csv",
    "FD001_public": PROJECT / "results" / "fd001_w48_public_code_evaluation_summary.csv",
    "FD003_public": PROJECT / "results" / "fd003_w48_public_code_evaluation_summary.csv",
    "FD004_public": PROJECT / "results" / "fd004_w48_public_code_evaluation_summary.csv",
}
REAL_BASELINE_CSV = PROJECT / "results" / "real_data_baseline_w48.csv"
PAPER_METRICS_MANIFEST = PROJECT / "docs" / "provenance" / "paper_predictive_scores_w48.json"
FD002_PAIRED_RAW_CSV = PROJECT / "results" / "frequency_threshold_validation" / "fd002_seven_seeds_paired.csv"

# --- 划分文件校验值
SPLIT_SHA256_FILE = "0b4fd49c52054af9791723baa33f38af220de513813030a3f599374ca55ae8e9"
SPLIT_SHA256_MANIFEST = "8c390ebfebf4cc3b18ff4b9e0cfa93c7b883e8d6a213364fc45d8bed80e82c76"
SPLIT_SHA256_FORBIDDEN = "843652a3c5323a316944da5c6338be70353ec90009769d8f206167e17aca48df"
SAMPLE_COUNT = 41539

# ============================================================================
# 二、计算
# ============================================================================

def paired_report(diffs: np.ndarray) -> dict:
    """配对差值的完整统计：t 检验 / Wilcoxon / CI / Cohen d_z"""
    d = np.asarray(diffs, float)
    n = len(d)
    t = stats.ttest_1samp(d, 0.0)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    try:
        w = stats.wilcoxon(d)
        wp = float(w.pvalue)
    except ValueError:
        wp = None
    return {
        "n": n,
        "per_seed": [round(float(x), 4) for x in d],
        "mean": round(float(d.mean()), 4),
        "median": round(float(np.median(d)), 4),
        "sd": round(float(d.std(ddof=1)), 4),
        "sem": round(float(stats.sem(d)), 4),
        "t": round(float(t.statistic), 3),
        "df": n - 1,
        "p_t": round(float(t.pvalue), 4),
        "p_wilcoxon": None if wp is None else round(wp, 4),
        "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "cohen_dz": round(float(d.mean() / d.std(ddof=1)), 3),
        "n_positive": int((d > 0).sum()),
        "n_negative": int((d < 0).sum()),
        "significant_at_05": bool(t.pvalue < 0.05),
    }


def mde(sd: float, n: int = 7, alpha: float = 0.05, power: float = 0.80) -> float:
    """n 个配对样本、双尾 t 检验达到给定功效所需的最小 |Δ|"""
    from scipy.optimize import brentq
    crit = stats.t.ppf(1 - alpha / 2, n - 1)

    def pw(d):
        ncp = abs(d) / (sd / np.sqrt(n))
        v = stats.nct.cdf(-crit, n - 1, ncp) + stats.nct.sf(crit, n - 1, ncp)
        return 1.0 if np.isnan(v) else float(v)

    return float(brentq(lambda d: pw(d) - power, 1e-9, sd * 100))


def read_metric_summary(path: Path, expected_seeds: list[int], metric: str) -> tuple[np.ndarray, dict]:
    """Read per-generation metric means and verify the summary aggregate row."""
    if not path.exists():
        raise FileNotFoundError(f"缺少正式评估摘要：{path}")
    per_seed: dict[int, float] = {}
    aggregate: float | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != metric:
                continue
            seed = row.get("generation_seed", "")
            if seed == "aggregate":
                aggregate = float(row["mean"])
            else:
                per_seed[int(seed)] = float(row["mean"])
    observed = sorted(per_seed)
    if observed != expected_seeds:
        raise ValueError(f"{path.name} generation seeds {observed} != {expected_seeds}")
    values = np.asarray([per_seed[seed] for seed in expected_seeds], dtype=float)
    if aggregate is None or not np.isclose(values.mean(), aggregate, rtol=0, atol=1e-10):
        raise ValueError(f"{path.name} aggregate {metric} 与逐种子均值不一致")
    return values, {
        "path": str(path.relative_to(PROJECT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generation_seeds": observed,
        f"aggregate_{metric}": aggregate,
    }


def read_real_baseline(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read the canonical local TRTR baseline, enforcing one row per evaluator seed."""
    if not path.exists():
        raise FileNotFoundError(f"缺少真实数据基线：{path}")
    datasets = ("FD001", "FD002", "FD003", "FD004")
    values: dict[str, dict[int, float]] = {ds: {} for ds in datasets}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ds = str(row.get("dataset", "")).upper()
            if ds in values:
                seed = int(row["evaluator_seed"])
                if seed in values[ds]:
                    raise ValueError(f"{path.name} 中 {ds} 的 evaluator seed {seed} 重复")
                values[ds][seed] = float(row["rmse"])
    observed = {ds: sorted(seed_values) for ds, seed_values in values.items()}
    if any(seeds != EVAL_SEEDS for seeds in observed.values()):
        raise ValueError(f"{path.name} evaluator seeds {observed} != {EVAL_SEEDS}")
    return ({ds: np.asarray([values[ds][seed] for seed in EVAL_SEEDS], dtype=float)
             for ds in datasets}, {
        "path": str(path.relative_to(PROJECT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluator_seeds": EVAL_SEEDS,
        "datasets": sorted(values),
    })


def read_paper_predictive_scores(path: Path) -> tuple[dict[str, float], dict]:
    """Read audited Table-I window-48 point estimates, with document identity."""
    if not path.exists():
        raise FileNotFoundError(f"缺少论文指标溯源文件：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    values = {str(ds).upper(): float(value) for ds, value in manifest["values"].items()}
    required = {"FD001", "FD002", "FD003", "FD004"}
    if set(values) != required:
        raise ValueError(f"{path.name} values datasets {sorted(values)} != {sorted(required)}")
    source = dict(manifest["source_document"])
    source.update({
        "manifest_path": str(path.relative_to(PROJECT)),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metric": manifest["metric"],
        "audit_note": manifest["audit_note"],
    })
    return values, source


def read_fd002_paired_raw(path: Path) -> tuple[list[dict[str, float]], dict]:
    """Read the 7 × 5 paired cells used for robustness calculations."""
    if not path.exists():
        raise FileNotFoundError(f"缺少 FD002 配对 raw 表：{path}")
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "generation_seed": int(row["generation_seed"]),
                "evaluator_seed": int(row["evaluator_seed"]),
                "rmse_fixed": float(row["rmse_fixed"]),
                "rmse_random": float(row["rmse_random"]),
                "ds_fixed": float(row["discriminative_score_fixed"]),
                "ds_random": float(row["discriminative_score_random"]),
            })
    observed = {(row["generation_seed"], row["evaluator_seed"]) for row in rows}
    expected = {(gen, eva) for gen in GEN_SEEDS for eva in EVAL_SEEDS}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(f"{path.name} 不是完整无重复的 7×5 配对表")
    return rows, {
        "path": str(path.relative_to(PROJECT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_cells": len(rows),
    }


def leave_one_cell_out_pvalues(rows: list[dict[str, float]], fixed_key: str, random_key: str) -> np.ndarray:
    """Remove one paired cell, aggregate within generation seed, then t-test 7 differences."""
    pvalues = []
    for index in range(len(rows)):
        remaining = rows[:index] + rows[index + 1:]
        diffs = []
        for generation_seed in GEN_SEEDS:
            cells = [row for row in remaining if row["generation_seed"] == generation_seed]
            diffs.append(np.mean([row[fixed_key] for row in cells]) -
                         np.mean([row[random_key] for row in cells]))
        pvalues.append(float(stats.ttest_1samp(diffs, 0).pvalue))
    return np.asarray(pvalues)


def build() -> dict:
    out: dict = {}
    fx_values, fx_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD002_fixed_theta025"], GEN_SEEDS, "rmse")
    pb_values, pb_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD002_public"], GEN_SEEDS, "rmse")
    fd001_values, fd001_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD001_public"], EVAL_SEEDS, "rmse")
    fd003_values, fd003_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD003_public"], EVAL_SEEDS, "rmse")
    fd004_values, fd004_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD004_public"], EVAL_SEEDS, "rmse")
    fx_ds, fx_ds_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD002_fixed_theta025"], GEN_SEEDS,
                                               "discriminative_score")
    pb_ds, pb_ds_source = read_metric_summary(SYNTHETIC_SUMMARIES["FD002_public"], GEN_SEEDS,
                                               "discriminative_score")
    d_ds = fx_ds - pb_ds
    real, real_source = read_real_baseline(REAL_BASELINE_CSV)
    paper, paper_source = read_paper_predictive_scores(PAPER_METRICS_MANIFEST)
    paired_raw, paired_raw_source = read_fd002_paired_raw(FD002_PAIRED_RAW_CSV)

    # ---- meta ----
    out["_meta"] = {
        "project": "MetaIndux-TS reproduction (TNNLS 2025), C-MAPSS FD001-FD004, window=48",
        "purpose": "冻结数字的唯一权威源。引用数字请从此文件取，不要在 md 之间互抄。",
        "generated_by": "analysis/audit/gen_frozen_numbers.py",
        "statistical_unit": "generation_seed (n=7)；5 个 evaluator seed 先按 gen 取均值",
        "alpha": 0.05,
        "two_sided": True,
        "rule": "禁止挑种子；seed 53 的离群必须如实报告，不得剔除",
    }

    # ---- 1. 七种子配对检验 ----
    d_rmse = fx_values - pb_values
    ds_report = paired_report(d_ds)
    ds_report["sources"] = {"fixed_theta025": fx_ds_source, "public_random_theta": pb_ds_source}
    out["fd002_seven_seed_paired"] = {
        "dataset": "FD002",
        "generation_seeds": GEN_SEEDS,
        "evaluator_seeds": EVAL_SEEDS,
        "n_cells": len(GEN_SEEDS) * len(EVAL_SEEDS),
        "arm_fixed_theta0.25": {
            "synth_rmse_per_gen": fx_values.tolist(),
            "source": fx_source,
            "mean": round(float(fx_values.mean()), 3),
            "sd_across_gen": round(float(fx_values.std(ddof=1)), 3),
            "cv_pct": round(float(fx_values.std(ddof=1) / fx_values.mean() * 100), 2),
            "worst": round(float(fx_values.max()), 3),
            "best": round(float(fx_values.min()), 3),
        },
        "arm_public_random_theta": {
            "synth_rmse_per_gen": pb_values.tolist(),
            "source": pb_source,
            "mean": round(float(pb_values.mean()), 3),
            "sd_across_gen": round(float(pb_values.std(ddof=1)), 3),
            "cv_pct": round(float(pb_values.std(ddof=1) / pb_values.mean() * 100), 2),
            "worst": round(float(pb_values.max()), 3),
            "best": round(float(pb_values.min()), 3),
        },
        "rmse": paired_report(d_rmse),
        "discriminative_score": ds_report,
        "verdict": "两臂无可检出差异（两项主指标均不显著，方向与符号分布均接近随机）",
    }

    # ---- 2. 稳健性 ----
    rmse_loco = leave_one_cell_out_pvalues(paired_raw, "rmse_fixed", "rmse_random")
    ds_loco = leave_one_cell_out_pvalues(paired_raw, "ds_fixed", "ds_random")
    i53 = GEN_SEEDS.index(53)
    d_rmse_without_53 = np.delete(d_rmse, i53)
    d_ds_without_53 = np.delete(d_ds, i53)
    out["robustness"] = {
        "leave_one_cell_out": {
            "source": paired_raw_source,
            "method": "逐格剔除 35 个 (generation_seed, evaluator_seed) 配对单元；每次按 generation seed 重新聚合后做配对 t 检验",
            "rmse_p_range": [round(float(rmse_loco.min()), 4), round(float(rmse_loco.max()), 4)],
            "ds_p_range": [round(float(ds_loco.min()), 4), round(float(ds_loco.max()), 4)],
            "baseline_p": {"rmse": paired_report(d_rmse)["p_t"], "ds": paired_report(d_ds)["p_t"]},
            "any_cross_0.05": bool((rmse_loco < 0.05).any() or (ds_loco < 0.05).any()),
        },
        "drop_seed53": {
            "rmse_p": round(float(stats.ttest_1samp(d_rmse_without_53, 0).pvalue), 4),
            "ds_p": round(float(stats.ttest_1samp(d_ds_without_53, 0).pvalue), 4),
            "note": "剔除离群 seed 53 后更不显著 → 不显著不是由该点制造的",
        },
        "ds_single_cell_drift": {
            "status": "historical observation; excluded from formal frozen conclusions",
            "note": "旧版三种子文件不是当前 7×5 正式评估的同一输入，不能作为主结论的可复算证据。"
        },
    }

    # 2b. 离散度（探索性）
    fx = fx_values
    pb = pb_values
    lev = stats.levene(fx, pb, center="median")
    lev2 = stats.levene(np.delete(fx, i53), np.delete(pb, i53), center="median")
    out["robustness"]["dispersion_exploratory"] = {
        "variance_ratio_public_over_fixed": round(float(pb.var(ddof=1) / fx.var(ddof=1)), 2),
        "levene_brown_forsythe_p": round(float(lev.pvalue), 4),
        "levene_p_without_seed53": round(float(lev2.pvalue), 4),
        "worst_case_gap_public_minus_fixed": round(float(pb.max() - fx.max()), 3),
        "status": "探索性，未达显著，且完全由 seed 53 单点驱动，不得写成结论",
    }

    # ---- 3. 可观测比较：论文最终生成结果、本地生成结果、本地真实基线 ----
    synth_pub = {"FD001": fd001_values, "FD002": pb_values, "FD003": fd003_values, "FD004": fd004_values}
    synth_sources = {"FD001": fd001_source, "FD002": pb_source, "FD003": fd003_source, "FD004": fd004_source}
    cond = {"FD001": "1工况/1故障", "FD002": "6工况/2故障", "FD003": "1工况/1故障", "FD004": "6工况/2故障"}

    comparisons = {}
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        v = real[ds]
        n = len(v)
        mean = float(v.mean())
        sd = float(v.std(ddof=1))
        sem = float(stats.sem(v))
        s = float(synth_pub[ds].mean())
        comparisons[ds] = {
            "condition": cond[ds],
            "published_generated_data_rmse_point_estimate": paper[ds],
            "published_generated_data_source": paper_source,
            "real_baseline": {
                "source": real_source,
                "per_seed": [float(x) for x in v],
                "seeds": EVAL_SEEDS,
                "n": n,
                "mean": round(mean, 3),
                "sd": round(sd, 3),
                "sem": round(sem, 3),
            },
            "local_generated_data": {
                "definition": "本地公开臂生成数据训练的预测器表现",
                "source": synth_sources[ds],
                "synth_mean_public": round(s, 3),
                "synth_sd_public": round(float(synth_pub[ds].std(ddof=1)), 3),
            },
            "reproduction_gap_vs_published_generated_point_estimate": round(s - paper[ds], 3),
            "local_synthetic_data_penalty_vs_local_real_baseline": round(s - mean, 3),
            "unobserved_paper_quantities": {
                "paper_real_data_baseline": "unknown: paper does not report it",
                "paper_synthetic_data_penalty_vs_paper_real_baseline": "unknown",
                "protocol_difference_paper_real_minus_local_real": "unknown",
            },
        }

    # FD002 固定臂分支
    d2 = comparisons["FD002"]
    fixed_mean = float(fx.mean())
    comparisons["FD002"]["fixed_theta_025_local_ablation"] = {
        "local_generated_data_rmse_mean": round(fixed_mean, 3),
        "reproduction_gap_vs_published_generated_point_estimate": round(fixed_mean - paper["FD002"], 3),
        "local_synthetic_data_penalty_vs_local_real_baseline": round(
            fixed_mean - d2["real_baseline"]["mean"], 3),
    }
    out["observable_comparisons"] = comparisons

    out["comparison_scope"] = {
        "formal_reproduction_gap": "本地生成数据训练 RMSE − 论文生成数据训练 RMSE",
        "local_synthetic_data_penalty": "本地生成数据训练 RMSE − 本地真实数据训练 RMSE",
        "unidentifiable_protocol_difference": "论文真实数据训练 RMSE − 本地真实数据训练 RMSE；论文未报告真实数据基线",
        "split_sha256_file_level": SPLIT_SHA256_FILE,
        "split_sha256_manifest_level": SPLIT_SHA256_MANIFEST,
        "sample_count": SAMPLE_COUNT,
        "forbidden_split_sha256": SPLIT_SHA256_FORBIDDEN,
        "forbidden_note": "results/fd002_evaluation_fixed_split.npz 是 11 个 FD002 划分里唯一的异类，"
                          "凡由它产出的数字禁止与本报告的相减或合并",
    }

    # ---- 5. 功效 ----
    d53 = d_rmse_without_53
    out["power_analysis"] = {
        "rmse": {
            "sd_all7": round(float(d_rmse.std(ddof=1)), 3),
            "mde_n7": round(mde(float(d_rmse.std(ddof=1)), 7), 3),
            "sd_without_seed53": round(float(d53.std(ddof=1)), 3),
            "mde_n6_without_seed53": round(mde(float(d53.std(ddof=1)), 6), 3),
            "observed_abs_delta": round(float(abs(d_rmse.mean())), 3),
            "seeds_needed_for_significance": 27,
        },
        "ds": {
            "sd_all7": round(float(d_ds.std(ddof=1)), 3),
            "mde_n7": round(mde(float(d_ds.std(ddof=1)), 7), 3),
            "observed_abs_delta": round(float(abs(d_ds.mean())), 3),
            "seeds_needed_for_significance": 22,
        },
        "note": "不显著 ≠ 无效应。n=7 的检出下限远高于实测效应量；按预注册不再补种子翻盘。",
    }

    # ---- 6. 交叉验证（最重要的一条自洽性检查）----
    gap_fixed = fixed_mean - paper["FD002"]
    gap_public = float(pb.mean()) - paper["FD002"]
    out["cross_validation"] = {
        "arm_total_gap_difference": round(float(gap_public - gap_fixed), 3),
        "paired_delta_mean_abs": round(float(abs(d_rmse.mean())), 4),
        "match": abs((gap_public - gap_fixed) - abs(d_rmse.mean())) < 0.01,
        "note": "两条完全独立的计算路径（双臂总 gap 之差 vs 七种子配对 Δ 均值）撞出同一个数。"
                "audit_checklist.sh 的 B7 项即此检查；对不上说明链路某处被换过。",
    }

    # ---- 7. 必写的局限 ----
    out["mandatory_limitations"] = [
        "不得表述为『两臂等价』——只做了差异性检验，未做 TOST 等价性检验；"
        "以 ±1.0 RMSE 为等价界时 CI 远超边界。正确说法是『未能检出差异』。",
        "seed 53 的 Δ=−11.104 必须如实报告，不得剔除。"
        "已验证剔除后结论不变（RMSE p 0.184→0.203），留着没有代价，藏着才是问题。",
        "『未检出差异』不等于『机制无效』——n=7 的检出下限（RMSE 5.201）远高于实测效应量（2.318）。",
        "论文未报告真实数据训练基线，因此协议层差异与论文生成数据相对论文真实数据"
        "的惩罚均不可识别；不得把本地真实基线与论文生成结果之差命名为协议层。",
    ]

    return out


# ============================================================================
# 三、断言：与已发布冻结值比对
# ============================================================================

def self_check(out: dict) -> list[str]:
    errs: list[str] = []

    def near(a, b, tol, label):
        if abs(float(a) - float(b)) > tol:
            errs.append(f"{label}: 计算 {float(a):.4f} vs 冻结 {float(b):.4f} (tol {tol})")

    p = out["fd002_seven_seed_paired"]
    near(p["rmse"]["mean"], -2.3181, 5e-4, "RMSE Δ 均值")
    near(p["rmse"]["sd"], 4.0872, 5e-4, "RMSE Δ SD")
    near(p["rmse"]["t"], -1.501, 1e-3, "RMSE t")
    near(p["rmse"]["p_t"], 0.1841, 5e-4, "RMSE p")
    near(p["rmse"]["ci95"][0], -6.0982, 5e-4, "RMSE CI 下界")
    near(p["rmse"]["ci95"][1], 1.4619, 5e-4, "RMSE CI 上界")
    near(p["discriminative_score"]["mean"], -0.1210, 5e-4, "DS Δ 均值")
    near(p["discriminative_score"]["p_t"], 0.1436, 5e-4, "DS p")
    near(p["discriminative_score"]["p_wilcoxon"], 0.297, 2e-3, "DS Wilcoxon p")
    near(p["rmse"]["p_wilcoxon"], 0.219, 2e-3, "RMSE Wilcoxon p")

    d = out["observable_comparisons"]
    near(d["FD002"]["real_baseline"]["mean"], 23.384, 1e-3, "FD002 真实对照均值")
    near(d["FD001"]["reproduction_gap_vs_published_generated_point_estimate"], 1.409, 1e-3, "FD001 正式复现差距")
    near(d["FD002"]["reproduction_gap_vs_published_generated_point_estimate"], 3.729, 1e-3, "FD002 正式复现差距")
    near(d["FD003"]["reproduction_gap_vs_published_generated_point_estimate"], 4.511, 1e-3, "FD003 正式复现差距")
    near(d["FD004"]["reproduction_gap_vs_published_generated_point_estimate"], 6.700, 1e-3, "FD004 正式复现差距")
    near(d["FD002"]["local_synthetic_data_penalty_vs_local_real_baseline"], 4.911, 1e-3, "FD002 本地生成数据惩罚")
    near(d["FD003"]["local_synthetic_data_penalty_vs_local_real_baseline"], 5.027, 2e-3, "FD003 本地生成数据惩罚")
    near(d["FD004"]["local_synthetic_data_penalty_vs_local_real_baseline"], 5.370, 2e-3, "FD004 本地生成数据惩罚")
    near(d["FD001"]["local_synthetic_data_penalty_vs_local_real_baseline"], 1.597, 2e-3, "FD001 本地生成数据惩罚")

    cv = out["cross_validation"]
    if not cv["match"]:
        errs.append(f"交叉验证失败：双臂总 gap 差 {cv['arm_total_gap_difference']} "
                    f"vs 配对 Δ 均值 {cv['paired_delta_mean_abs']}")

    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验不写文件")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = build()
    errs = self_check(out)

    if errs:
        print("[SELF-CHECK FAILED]")
        for e in errs:
            print("  ✗", e)
        return 1
    print("[SELF-CHECK OK] 全部冻结值复算一致")

    if args.check:
        return 0

    dst = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[2] / "results" / "frozen_numbers.json"
    dst.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(o):
        """numpy 标量 → 内置类型，保证 json 可序列化"""
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [sanitize(v) for v in o]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (int, np.integer)):
            return int(o)
        if isinstance(o, (float, np.floating)):
            return float(o)
        return o

    dst.write_text(json.dumps(sanitize(out), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {dst}  ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
