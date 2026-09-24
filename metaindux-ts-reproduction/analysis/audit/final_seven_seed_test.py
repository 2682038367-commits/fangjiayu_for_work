#!/usr/bin/env python3
"""FD002 七种子配对检验（决定是否收尾的那一步）。

对比：固定 θ=0.25 臂  vs  公开代码随机 θ 臂
单位：generation_seed（n=7），5 个 evaluator_seed 先聚合成一个点，不做伪重复

预注册（在看到 7 种子结果之前已声明）：
  - 双主指标并列：RMSE、判别分数 DS
  - 双尾配对 t 检验，α = 0.05
  - n = 7，不做多重比较校正，两个指标都报告原始 p

用法：
  .venv/bin/python final_seven_seed_test.py
  .venv/bin/python final_seven_seed_test.py --csv <指定 csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "results/frequency_threshold_validation"

# 预注册的功效分析预测值（基于旧 3 种子的 SD 外推，务必与实际 SD 对照）
PREREG = {
    "rmse": {"sd_hat": 1.5405, "mde80": 1.425, "obs_delta": -1.0067},
    "ds":   {"sd_hat": 0.197,  "mde80": 0.182, "obs_delta": -0.171},
}

# METRICS 的首项是 agg / PREREG 内部用的短键（与聚合字典一致），不可改
METRICS = [
    ("rmse", "RMSE", "越小越好", "#c62828"),
    ("ds", "判别分数 DS", "越小越好", "#1565c0"),
]


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
def pick_col(fieldnames, cands, what):
    """在候选列名里挑一个存在的；挑不到就报出实际列名让人排查。"""
    low = {f.lower(): f for f in fieldnames}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise SystemExit(
        f"[FATAL] 找不到{what}列。试过候选 {cands}\n"
        f"        csv 实际列名 = {fieldnames}\n"
        f"        → 把这一行贴回来，我加进候选列表。"
    )


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        fields = rd.fieldnames or []
    col_map = {}
    for key, cands, what in [
        ("gen", ["generation_seed", "gen_seed", "seed"], "generation_seed"),
        ("eval", ["evaluator_seed", "eval_seed"], "evaluator_seed"),
        ("rmse_pub", ["rmse_random", "rmse_public", "public_rmse"], "公开臂 RMSE"),
        ("rmse_fix", ["rmse_fixed", "fixed_rmse"], "固定臂 RMSE"),
        ("ds_pub", ["discriminative_score_random", "ds_random",
                    "discriminative_score_public", "ds_public"], "公开臂 DS"),
        ("ds_fix", ["discriminative_score_fixed", "ds_fixed"], "固定臂 DS"),
    ]:
        col_map[key] = pick_col(fields, cands, what)
    data = []
    for r in rows:
        data.append({
            "gen": int(r[col_map["gen"]]),
            "eval": int(r[col_map["eval"]]),
            "rmse_pub": float(r[col_map["rmse_pub"]]),
            "rmse_fix": float(r[col_map["rmse_fix"]]),
            "ds_pub": float(r[col_map["ds_pub"]]),
            "ds_fix": float(r[col_map["ds_fix"]]),
        })
    return data, fields


def mde(sd, n=7, alpha=0.05, power=0.80):
    """n=7 双尾配对 t 检验，要达到指定功效所需的最小均值差绝对值。"""
    from scipy.optimize import brentq
    crit = stats.t.ppf(1 - alpha / 2, n - 1)

    def pw(d):
        ncp = abs(d) / (sd / np.sqrt(n))
        p = stats.nct.cdf(-crit, n - 1, ncp) + stats.nct.sf(crit, n - 1, ncp)
        # 极端非中心参数下 nct 会返回 NaN，此时功效饱和，按 1 处理
        return 1.0 if np.isnan(p) else float(p)

    return brentq(lambda d: pw(d) - power, 1e-6, sd * 50)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    cli = ap.parse_args()

    if cli.csv:
        path = Path(cli.csv)
    else:
        cands = sorted(OUT_DIR.glob("*seven_seeds*paired*.csv"))
        if cands:
            path = cands[-1]
        else:
            path = OUT_DIR / "fd002_fixed_025_paired_comparison.csv"
            print("⚠ 未找到 seven_seeds 配对表，退回旧 3 种子存档（只会有 3 对）")
    if not path.is_file():
        sys.exit(f"[FATAL] 找不到 {path}")

    rows, fields = load(path)
    print(f"读入 {path.name}   行数 = {len(rows)}")
    print(f"列 {fields}\n")

    # ---------------- 1. 结构守卫 ----------------
    print("=" * 78)
    print("1. 结构守卫（不合格就不能做 7 对检验）")
    print("=" * 78)
    gens = sorted({r["gen"] for r in rows})
    evals = sorted({r["eval"] for r in rows})
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  [OK]   " if cond else "  [FAIL] ") + msg)
        if not cond:
            ok = False

    chk(set(gens) == {3, 13, 23, 33, 43, 53, 63}, f"generation_seeds = {gens}（7 个）")
    chk(len(evals) == 5, f"evaluator_seeds = {evals}（5 个）")
    chk(len(rows) == 35, f"行数 = {len(rows)}（35 = 7×5）")
    dup = len(rows) - len({(r["gen"], r["eval"]) for r in rows})
    chk(dup == 0, f"无重复 (gen, eval) 组合，重复数 = {dup}")
    per = {g: sum(1 for r in rows if r["gen"] == g) for g in gens}
    chk(all(v == 5 for v in per.values()), f"每个 gen 均有 5 个 eval：{per}")
    if not ok:
        sys.exit("\n结构检查未通过 → 不要进统计，先排查。")

    # ---------------- 2. 聚合成 7 个点 ----------------
    print()
    print("=" * 78)
    print("2. 按 generation_seed 聚合（5 个 evaluator 先取均值）")
    print("   统计单位必须是 generation_seed，按 35 行算属伪重复")
    print("=" * 78)
    agg = {}
    for g in gens:
        sub = [r for r in rows if r["gen"] == g]
        agg[g] = {
            "rmse_pub": np.mean([r["rmse_pub"] for r in sub]),
            "rmse_fix": np.mean([r["rmse_fix"] for r in sub]),
            "ds_pub": np.mean([r["ds_pub"] for r in sub]),
            "ds_fix": np.mean([r["ds_fix"] for r in sub]),
            "rmse_pub_sd": np.std([r["rmse_pub"] for r in sub], ddof=1),
            "rmse_fix_sd": np.std([r["rmse_fix"] for r in sub], ddof=1),
            "ds_pub_sd": np.std([r["ds_pub"] for r in sub], ddof=1),
            "ds_fix_sd": np.std([r["ds_fix"] for r in sub], ddof=1),
        }
    hdr = (f"{'seed':>5} | {'公开RMSE':>9} {'固定RMSE':>9} {'ΔRMSE':>8} | "
           f"{'公开DS':>8} {'固定DS':>8} {'ΔDS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for g in gens:
        a = agg[g]
        dr = a["rmse_fix"] - a["rmse_pub"]
        dd = a["ds_fix"] - a["ds_pub"]
        print(f"{g:>5} | {a['rmse_pub']:>9.4f} {a['rmse_fix']:>9.4f} {dr:>+8.4f} | "
              f"{a['ds_pub']:>8.4f} {a['ds_fix']:>8.4f} {dd:>+8.4f}")
    print("-" * len(hdr))
    for key, name, _better, _c in METRICS:
        pub = np.array([agg[g][f"{key}_pub"] for g in gens])
        fix = np.array([agg[g][f"{key}_fix"] for g in gens])
        print(f"{name:>10}  均值 {pub.mean():.4f} → {fix.mean():.4f}"
              f"   （固定臂 {'更优' if fix.mean() < pub.mean() else '更差'}"
              f" {abs(fix.mean()-pub.mean()):.4f}）")

    # ---------------- 3. 配对检验 ----------------
    print()
    print("=" * 78)
    print("3. 七对配对检验（双尾，α=0.05，单位 = generation_seed，n=7）")
    print("=" * 78)
    results = {}
    for key, name, better, _c in METRICS:
        pub = np.array([agg[g][f"{key}_pub"] for g in gens])
        fix = np.array([agg[g][f"{key}_fix"] for g in gens])
        d = fix - pub
        t = stats.ttest_rel(fix, pub)
        w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=stats.sem(d))
        dz = d.mean() / d.std(ddof=1)
        sd = d.std(ddof=1)
        need = mde(sd)
        sh = stats.shapiro(d)
        results[key] = dict(
            name=name, pub=pub, fix=fix, d=d, t=t.statistic, p=t.pvalue,
            w=w.statistic, wp=w.pvalue, ci=ci, dz=dz, sd=sd, need=need,
            shp=sh.pvalue, better=better,
        )
        print(f"\n  ── {name}（{better}）──")
        print(f"     逐种子 Δ : {[f'{v:+.4f}' for v in d]}")
        print(f"     Δ 均值 {d.mean():+.4f}   SD {sd:.4f}   SEM {stats.sem(d):.4f}")
        print(f"     配对 t  = {t.statistic:+.3f}   p = {t.pvalue:.4f}   df = 6")
        print(f"     95% CI  = [{ci[0]:+.4f}, {ci[1]:+.4f}]"
              f"   {'含 0 → 不显著' if ci[0] < 0 < ci[1] else '不含 0 → 显著'}")
        print(f"     Cohen d_z = {dz:+.3f}    Wilcoxon W = {w.statistic:.1f}, p = {w.pvalue:.4f}")
        print(f"     差值正态性 Shapiro p = {sh.pvalue:.4f}"
              f"   {'（正态假设可疑，以 Wilcoxon 为准）' if sh.pvalue < 0.05 else ''}")
        print(f"     本 SD 下 n=7 要显著需 |Δ| ≥ {need:.4f}，实测 |Δ| = {abs(d.mean()):.4f}"
              f"   → {'够' if abs(d.mean()) >= need else '不够'}")

    # ---------------- 4. 与预注册对照 ----------------
    print()
    print("=" * 78)
    print("4. 与预注册对照（防止把探索结果包装成验证）")
    print("=" * 78)
    print(f"{'指标':<12}{'预注册 SD':>11}{'实测 SD':>10}{'SD 比':>8}"
          f"{'预注册需|Δ|':>13}{'实测|Δ|':>10}{'判定':>16}")
    print("-" * 82)
    for key, name, _b, _c in METRICS:
        r = results[key]
        sd_hat = PREREG[key]["sd_hat"]
        ratio = r["sd"] / sd_hat
        verdict = ("命中预注册" if abs(r["d"].mean()) >= PREREG[key]["mde80"]
                   else "低于检出阈值")
        print(f"{name:<12}{sd_hat:>11.4f}{r['sd']:>10.4f}{ratio:>8.2f}×"
              f"{PREREG[key]['mde80']:>13.4f}{abs(r['d'].mean()):>10.4f}{verdict:>16}")
    print("-" * 82)
    print("  注：预注册 SD 由旧 3 种子的 Δ SD 外推而来。若实测 SD 明显不同，")
    print("      说明 3 种子那批低估/高估了方差，预注册的功效预测失效，需如实披露。")

    # ---------------- 5. 结论措辞 ----------------
    print()
    print("=" * 78)
    print("5. 可直接写进论文的措辞")
    print("=" * 78)
    for key, name, better, _c in METRICS:
        r = results[key]
        sig = r["p"] < 0.05
        if sig:
            direction = "显著优于" if r["d"].mean() < 0 else "显著劣于"
            s = (f"{name}：固定 θ=0.25 {direction}公开随机 θ "
                 f"(Δ = {r['d'].mean():+.4f}, 95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}], "
                 f"t(6) = {r['t']:+.2f}, p = {r['p']:.4f}, d_z = {r['dz']:+.2f})。")
        else:
            s = (f"{name}：固定 θ=0.25 与公开随机 θ 无显著差异 "
                 f"(Δ = {r['d'].mean():+.4f}, 95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}], "
                 f"t(6) = {r['t']:+.2f}, p = {r['p']:.3f})；"
                 f"在观测到的 SD = {r['sd']:.3f} 下，n = 7 仅能检出 |Δ| ≥ {r['need']:.3f} 的差异，"
                 f"故本结果应解释为『未检出』而非『无差异』。")
        print("  ▸ " + s)
        print()

    _write_report(path, gens, agg, results)
    _plot(gens, agg, results)
    print(f"报告 → fd002_seven_seed_paired_report.md")
    print(f"图   → fd002_seven_seed_paired_plot.png")


def _write_report(path, gens, agg, results):
    lines = ["# FD002 固定 θ=0.25 vs 公开随机 θ：七种子配对检验", ""]
    lines += [f"数据源：`{path}`（{len(gens)} 生成种子 × 5 评估种子）", ""]
    lines += ["单位 = generation_seed（n=7）。5 个 evaluator_seed 先聚合成一点，"
              "避免伪重复。", ""]
    lines += ["## 逐种子结果", ""]
    lines += ["| gen seed | 公开 RMSE | 固定 RMSE | ΔRMSE | 公开 DS | 固定 DS | ΔDS |"]
    lines += ["|---:|---:|---:|---:|---:|---:|---:|"]
    for g in gens:
        a = agg[g]
        lines.append(
            f"| {g} | {a['rmse_pub']:.4f} | {a['rmse_fix']:.4f} | "
            f"{a['rmse_fix']-a['rmse_pub']:+.4f} | {a['ds_pub']:.4f} | "
            f"{a['ds_fix']:.4f} | {a['ds_fix']-a['ds_pub']:+.4f} |")
    lines += [""]
    lines += ["## 配对检验", ""]
    lines += ["| 指标 | Δ 均值 | SD | t(6) | p | 95% CI | Cohen d_z | Wilcoxon p | 判定 |"]
    lines += ["|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for key, name, _b, _c in METRICS:
        r = results[key]
        lines.append(f"| {name} | {r['d'].mean():+.4f} | {r['sd']:.4f} | {r['t']:+.3f} | "
                     f"{r['p']:.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                     f"{r['dz']:+.3f} | {r['wp']:.4f} | "
                     f"{'显著' if r['p'] < 0.05 else '不显著'} |")
    lines += [""]
    lines += ["## 检出能力（事后）", ""]
    for key, name, _b, _c in METRICS:
        r = results[key]
        lines.append(f"- {name}：在实测 SD = {r['sd']:.4f} 下，n=7、α=0.05、power=0.80 "
                     f"需 |Δ| ≥ {r['need']:.4f}；实测 |Δ| = {abs(r['d'].mean()):.4f}。")
    lines += [""]
    lines += ["不显著的情形必须写成『未检出（under-powered）』，"
              "不能写成『两者无差异』。", ""]
    Path("fd002_seven_seed_paired_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


def _plot(gens, agg, results):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 中文字体（与 theta_rmse_analysis.py 保持同一套，避免导出方框）
    for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(_p):
            font_manager.fontManager.addfont(_p)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130,
                         "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                         "axes.unicode_minus": False})

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.32)

    for k, (key, name, _b, col) in enumerate(METRICS):
        ax = fig.add_subplot(gs[k])
        for i, g in enumerate(gens):
            a = agg[g]
            pub, fix = a[f"{key}_pub"], a[f"{key}_fix"]
            ax.plot([0, 1], [pub, fix], marker="o", lw=1.6, ms=7,
                    color=col, alpha=0.75, zorder=3)
        # 两侧端点标签都要避让：真实数据里 s13/s23 的固定臂值只差 0.23，
        # 直接用原始 y 会叠在一起
        span = max(agg[g][f"{key}_fix"] for g in gens) - \
            min(agg[g][f"{key}_fix"] for g in gens)
        need = span * 0.058 if span > 0 else 0.0
        last = None
        for i in sorted(range(len(gens)),
                        key=lambda j: agg[gens[j]][f"{key}_fix"]):
            g = gens[i]
            yv = agg[g][f"{key}_fix"]
            if last is not None and yv - last < need:
                yv = last + need
            last = yv
            ax.annotate(f"s{g}", (1.05, yv), fontsize=8, va="center")
        spanp = max(agg[g][f"{key}_pub"] for g in gens) - \
            min(agg[g][f"{key}_pub"] for g in gens)
        needp = spanp * 0.058 if spanp > 0 else 0.0
        last = None
        for i in sorted(range(len(gens)),
                        key=lambda j: agg[gens[j]][f"{key}_pub"]):
            g = gens[i]
            d = agg[g][f"{key}_fix"] - agg[g][f"{key}_pub"]
            yv = agg[g][f"{key}_pub"]
            if last is not None and yv - last < needp:
                yv = last + needp
            last = yv
            ax.annotate(f"{d:+.2f}", (-0.06, yv), fontsize=8, ha="right",
                        va="center", color="#c62828" if d < 0 else "#333",
                        fontweight="bold" if abs(d) > results[key]["sd"] else "normal")
        mean_pub = np.mean([agg[g][f"{key}_pub"] for g in gens])
        mean_fix = np.mean([agg[g][f"{key}_fix"] for g in gens])
        ax.plot([0, 1], [mean_pub, mean_fix], marker="s", lw=3.2, ms=11,
                color="#2e7d32", zorder=4, label="七种子均值")
        ax.set_xlim(-0.42, 1.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["公开臂\n随机 θ", "固定臂\nθ=0.25"], fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        r = results[key]
        verdict = f"p = {r['p']:.3f}" + ("  ✓显著" if r["p"] < 0.05 else "  不显著")
        ax.set_title(f"{'A' if k == 0 else 'B'}. {name}逐种子配对\n"
                     f"Δ均值 {r['d'].mean():+.3f}, t(6) = {r['t']:+.2f}, {verdict}",
                     fontsize=10.5, loc="left")
        ax.grid(alpha=0.22, axis="y")
        ax.legend(fontsize=8, loc="upper right")

    # C: Δ 分布
    ax = fig.add_subplot(gs[2])
    for k, (key, name, _b, col) in enumerate(METRICS):
        d = results[key]["d"]
        ax.scatter([k] * len(d), d, s=70, color=col, ec="white", lw=1.1,
                   zorder=3, alpha=0.85)
        m = d.mean()
        ax.hlines(m, k - 0.22, k + 0.22, color=col, lw=2.6, zorder=4)
        ci = results[key]["ci"]
        ax.plot([k, k], ci, color=col, lw=1.4, alpha=0.8, zorder=4)
        ax.plot([k - 0.07, k + 0.07], [ci[0]] * 2, color=col, lw=1.4)
        ax.plot([k - 0.07, k + 0.07], [ci[1]] * 2, color=col, lw=1.4)
    ax.axhline(0, c="#999", lw=1.2, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([METRICS[0][1], METRICS[1][1]], fontsize=10)
    ax.set_ylabel("Δ（固定 − 公开）", fontsize=10)
    ax.set_title("C. 配对差及其 95% CI（跨过 0 = 不显著）",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=0.22, axis="y")

    fig.suptitle("FD002：固定 θ=0.25 vs 公开随机 θ（7 生成种子配对，窗口 48）",
                 fontsize=13, y=1.00)
    fig.savefig("fd002_seven_seed_paired_plot.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
