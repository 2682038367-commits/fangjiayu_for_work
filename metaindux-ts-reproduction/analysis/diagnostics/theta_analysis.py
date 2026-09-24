"""
RETIRED historical exploratory analysis (2026-09-24).

This file embeds checkpoint values and soft/STE experimental values by hand.
It is not a formal public-code reproduction input and must not regenerate a
current report.  Use theta_rmse_analysis.py for the CSV-backed public-code
analysis; use recorded mask_drop_ratio for any cross-mode mask claim.

从 checkpoint 读出的 θ 值 → 统计分析 + 绘图
==========================================================================
输入: 用户实测的 threshold_param (ups.0.2 = θ1, ups.1.2 = θ2)
输出: theta_scatter.html  +  控制台统计表
"""
raise SystemExit(
    "[RETIRED] theta_analysis.py contains hand-entered exploratory values; "
    "use theta_rmse_analysis.py instead."
)

import itertools
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================ 实测数据
# 公开臂（随机初始化后冻结）：seed -> (θ1, θ2)
PUBLIC = {
    3:  (0.3644869327545166, 0.4880968928337097),
    13: (0.2985737919807434, 0.49599048495292664),
    23: (0.0741274356842041, 0.49625810980796814),
    33: (0.06935897469520569, 0.41670921444892883),
    43: (0.28632768988609314, 0.4584011435508728),
    53: (0.2981482744216919, 0.04880553483963013),
    63: (0.27352333068847656, 0.18644553422927856),
}
# 固定臂 freq_threshold_025（FD002 三个种子，全部命中 0.25/0.25）
FIXED = {13: (0.25, 0.25), 23: (0.25, 0.25), 43: (0.25, 0.25)}

# 可学习臂的原始参数（未 clamp 的 logit / 原始实数）
SOFT = {
    3:  (-0.8097494840621948, -4.3301286697387695),
    13: (-1.204519271850586,  -4.130945682525635),
    23: (0.23161210119724274, -1.774895429611206),
    33: (-1.3639973402023315, -4.086968898773193),
    43: (0.023020537570118904, -2.8020777702331543),
    53: (-0.438871830701828,  -4.138935089111328),
    63: (-1.4909751415252686, -4.034384250640869),
}
STE = {
    3:  (0.30132338404655457, -1.6774977445602417),
    13: (0.4321889877319336,  0.6366026401519775),
    23: (-0.7684102058410645, -2.1629629135131836),
    33: (-0.9955490827560425, -2.023256301879883),
    43: (0.030580701306462288, -1.0221341848373413),
    53: (-0.9227482080459595,  1.08234703540802),
    63: (-0.054345931857824326, -1.0047533512115479),
}

# FD004 公开臂的 best_epoch（用户此前提供）
FD004_EPOCH = {3: 25, 13: 24, 23: 68, 33: 27, 43: 21}

# ============================================================ 工具
def perm_spearman(x, y, n_perm=None):
    """Spearman ρ + 精确/随机置换检验（双尾）"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho, _ = stats.spearmanr(x, y)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    n = len(x)
    if n_perm is None and n <= 8:
        cnt, tot = 0, 0
        for p in itertools.permutations(range(n)):
            tot += 1
            if abs(np.corrcoef(rx, ry[list(p)])[0, 1]) >= abs(rho) - 1e-12:
                cnt += 1
        return rho, cnt / tot, tot
    rng = np.random.default_rng(0)
    cnt = 0
    N = n_perm or 20000
    for _ in range(N):
        p = rng.permutation(n)
        if abs(np.corrcoef(rx, ry[p])[0, 1]) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / N, N


def sig(v):        # logit -> 概率
    return 1.0 / (1.0 + np.exp(-np.asarray(v, float)))


def soft_eff(v):   # soft_learnable_energy: normalized-energy cutoff
    return 0.5 * sig(v)


def ste_eff(v):    # binary_ste_energy: normalized-energy cutoff
    return np.log1p(np.exp(np.asarray(v, float)))


# ============================================================ 统计输出
L = []
P = lambda s: L.append(s)

P("=" * 84)
P("A. 公开臂 θ 的分布 —— 是否符合 U(0, 0.5)？")
P("=" * 84)
t1 = np.array([PUBLIC[s][0] for s in PUBLIC])
t2 = np.array([PUBLIC[s][1] for s in PUBLIC])
seeds = sorted(PUBLIC)
P(f"{'seed':>5} | {'θ1 (ups.0.2)':>14} | {'θ2 (ups.1.2)':>14} | {'均值':>8} | θ2−θ1")
P("-" * 84)
for s in seeds:
    a, b = PUBLIC[s]
    P(f"{s:>5} | {a:>14.4f} | {b:>14.4f} | {(a+b)/2:>8.4f} | {b-a:+.4f}")
P("-" * 84)
P(f"{'均值':>5} | {t1.mean():>14.4f} | {t2.mean():>14.4f} | {(t1+t2).mean()/2:>8.4f} |")
P(f"{'SD':>5} | {t1.std(ddof=1):>14.4f} | {t2.std(ddof=1):>14.4f} |")
P(f"理论 U(0,0.5): 均值 0.2500, SD {0.5/np.sqrt(12):.4f}")

allt = np.concatenate([t1, t2])
ks = stats.kstest(allt / 0.5, "uniform")
P("")
P(f"合并 14 个 θ 对 U(0,0.5) 做 KS 检验:  D = {ks.statistic:.4f},  p = {ks.pvalue:.3f}")
P("  → p > 0.05，无法拒绝『θ ~ U(0,0.5)』，抽样本身不异常。")
P("")
n_hi = int((t2 >= 0.4167).sum())
P(f"但 θ2 一侧偏高: 7 个种子里 {n_hi} 个 θ2 ≥ 0.4167 (上界 0.5 的 83%)")
P(f"  前五种子(3/13/23/33/43) θ2 ∈ [{t2[:5].min():.4f}, {t2[:5].max():.4f}]，全部 > 0.41")
P(f"  seed 53/63 的 θ2 = {PUBLIC[53][1]:.4f} / {PUBLIC[63][1]:.4f} —— 明显低")
P(f"  θ2 均值 {t2.mean():.4f} vs θ1 均值 {t1.mean():.4f}  vs  我们选定的固定值 0.25")

rho12, p12, np12 = perm_spearman(t1, t2)
P("")
P(f"θ1 与 θ2 的相关: Spearman ρ = {rho12:+.3f}, 置换 p = {p12:.3f} ({np12} 排列)")
P("  → 当前样本中未检测到两个模块参数的相关性；不等于确认统计独立。")

# ============================================================
P("")
P("=" * 84)
P("B. θ 与训练动力学：FD004 best_epoch（n=5，仅提示性）")
P("=" * 84)
sub = [s for s in seeds if s in FD004_EPOCH]
ep = np.array([FD004_EPOCH[s] for s in sub])
P(f"{'seed':>5} | {'θ1':>8} | {'θ2':>8} | {'θ均值':>8} | best_epoch")
P("-" * 84)
for s in sub:
    a, b = PUBLIC[s]
    P(f"{s:>5} | {a:>8.4f} | {b:>8.4f} | {(a+b)/2:>8.4f} | {FD004_EPOCH[s]:>10d}")
P("-" * 84)
for name, vec in [("θ1", [PUBLIC[s][0] for s in sub]),
                  ("θ2", [PUBLIC[s][1] for s in sub]),
                  ("θ均值", [(PUBLIC[s][0] + PUBLIC[s][1]) / 2 for s in sub])]:
    r, p, tot = perm_spearman(vec, ep)
    P(f"{name:<6} vs best_epoch:  Spearman ρ = {r:+.3f},  精确置换 p = {p:.3f}  ({tot} 排列)")
P("")
P("  ⚠ n=5 时双尾置换检验能达到的最小 p = 2/120 = 0.017（且需 |ρ|=1）。")
P("    ρ=−0.5 的 p ≈ 0.4，远不显著。方向『θ1 越小 → 需要越多 epoch』")
P("    只在 seed23 (θ1=0.074, epoch=68) 这一个点上成立，不能推广。")

# ============================================================
P("")
P("=" * 84)
P("C. 可学习臂：参数跑飞到负值区")
P("=" * 84)
P("公开臂 θ 恒在 (0, 0.5) 内；soft / binary-STE 臂的原始参数落到 [-4.33, 1.08]。")
P("源码确认的映射（Attetnion_temp.py::effective_threshold）：")
P("  soft 臂: 归一化能量阈值 = 0.5·sigmoid(θ)   （值域 (0, 0.5)）")
P("  STE 臂:  归一化能量阈值 = softplus(θ)      （值域 (0, +∞)）")
P("  注意：这两个值都不是丢弃比例；实际丢弃/衰减率必须从前向 mask 统计。")
P("")
P(f"{'seed':>5} | {'soft θ1':>9} {'0.5σ':>6} | {'soft θ2':>9} {'0.5σ':>6} | {'STE θ1':>9} {'softplus':>8} | {'STE θ2':>9} {'softplus':>8}")
P("-" * 84)
for s in seeds:
    a, b = SOFT[s]
    c, d = STE[s]
    P(f"{s:>5} | {a:>9.3f} {soft_eff(a):>6.3f} | {b:>9.3f} {soft_eff(b):>6.3f} | "
      f"{c:>9.3f} {ste_eff(c):>8.3f} | {d:>9.3f} {ste_eff(d):>8.3f}")
P("-" * 84)
s2 = np.array([soft_eff(SOFT[s][1]) for s in seeds])
s1 = np.array([soft_eff(SOFT[s][0]) for s in seeds])
P(f"soft: θ2 能量阈值中位数 {np.median(s2):.4f}（7 个里 {int((s2<0.01).sum())} 个 < 0.01）")
P(f"      θ1 能量阈值范围 [{s1.min():.3f}, {s1.max():.3f}]，跨度 {(s1.max()-s1.min()):.3f} —— 分散")
P(f"      原始 θ2 有 {int((np.array([SOFT[s][1] for s in seeds]) < -4).sum())} 个落在 [−4.33, −4.03]，几乎同一处")
P("")
P("  → θ2 的归一化能量阈值趋近于零；仅凭阈值数值不能量化保留比例，")
P("    也不能据此确认退化解。应使用采样前向记录的 mask_drop_ratio 验证。")

# ============================================================ 绘图
from matplotlib import font_manager
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130,
                     "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
                     "axes.unicode_minus": False})
fig = plt.figure(figsize=(14, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                       top=0.885, bottom=0.075, left=0.055, right=0.985)

# --- 图 A: θ1 × θ2 平面
ax = fig.add_subplot(gs[0, :2])
ax.add_patch(plt.Rectangle((0, 0), 0.5, 0.5, fill=False, ls="--", lw=1.2,
                           ec="#888", label="初始化域 U(0,0.5)²"))
ax.plot([0.25], [0.25], marker="s", ms=15, mfc="#e8f5e9", mec="#2e7d32",
        mew=2.2, ls="none", label="固定臂 θ=0.25/0.25（3 种子重合）")
for s in seeds:
    a, b = PUBLIC[s]
    ax.scatter(a, b, s=150, c="#1565c0", ec="white", lw=1.4, zorder=3)
    ax.annotate(f"seed{s}", (a, b), textcoords="offset points",
                xytext=(8, 6), fontsize=9, fontweight="bold")
ax.axhline(0.25, c="#2e7d32", lw=0.9, alpha=0.5)
ax.axvline(0.25, c="#2e7d32", lw=0.9, alpha=0.5)
ax.axhline(t2.mean(), c="#c62828", lw=1.3, ls=":",
           label=f"公开臂 θ2 均值 {t2.mean():.3f}")
ax.axvline(t1.mean(), c="#ef6c00", lw=1.3, ls=":",
           label=f"公开臂 θ1 均值 {t1.mean():.3f}")
ax.set_xlim(-0.03, 0.58); ax.set_ylim(-0.03, 0.58)
ax.set_xlabel("θ₁  (ups.0.2.fn.fn.threshold_param)  —— 分位数参数")
ax.set_ylabel("θ₂  (ups.1.2.fn.fn.threshold_param)")
ax.set_title("A. 公开臂逐种子抽到的 θ 对（FD002/003/004 同 seed 完全相同）\n"
             "θ₂ 系统性贴近上界（实际丢弃率需由前向 mask 核验）",
             fontsize=10.5, loc="left")
ax.legend(fontsize=8, loc="lower right", framealpha=0.95)

# --- 图 B: θ 条带
ax = fig.add_subplot(gs[0, 2])
ax.axvspan(0, 0.5, color="#f0f0f0", zorder=0)
for i, (s, (a, b)) in enumerate(sorted(PUBLIC.items())):
    ax.plot([a, b], [i, i], c="#bbb", lw=1.6, zorder=1)
    ax.scatter([a], [i], s=70, c="#ef6c00", zorder=2, label="θ₁" if i == 0 else None)
    ax.scatter([b], [i], s=70, c="#1565c0", zorder=2, label="θ₂" if i == 0 else None)
ax.axvline(0.25, c="#2e7d32", lw=2, label="固定 0.25")
ax.set_yticks(range(len(PUBLIC)))
ax.set_yticklabels([f"seed{s}" for s in sorted(PUBLIC)], fontsize=8)
ax.set_xlim(-0.02, 0.56)
ax.set_xlabel("θ（目标分位数；不是实测丢弃率）")
ax.set_title(f"B. 逐种子 θ₁/θ₂\nKS vs U(0,0.5): D={ks.statistic:.3f}, p={ks.pvalue:.2f}",
             fontsize=10, loc="left")
ax.legend(fontsize=8)

# --- 图 C: θ vs best_epoch
ax = fig.add_subplot(gs[1, 0])
for name, idx, c in [("θ₁", 0, "#ef6c00"), ("θ₂", 1, "#1565c0")]:
    xs = [PUBLIC[s][idx] for s in sub]
    ax.scatter(xs, ep, s=90, c=c, label=name, zorder=3)
    z = np.polyfit(xs, ep, 1)
    xs2 = np.linspace(min(xs) - 0.02, max(xs) + 0.02, 20)
    ax.plot(xs2, np.polyval(z, xs2), c=c, lw=1.3, alpha=0.6)
for s, x, y in zip(sub, [PUBLIC[s][0] for s in sub], ep):
    ax.annotate(f"{s}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
r1, p1, _ = perm_spearman([PUBLIC[s][0] for s in sub], ep)
r2, p2, _ = perm_spearman([PUBLIC[s][1] for s in sub], ep)
ax.set_xlabel("θ"); ax.set_ylabel("FD004 best_epoch")
ax.set_title(f"C. θ vs best_epoch (n=5)\nθ₁: ρ={r1:+.2f} (p={p1:.2f})   θ₂: ρ={r2:+.2f} (p={p2:.2f})",
             fontsize=10, loc="left")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

# --- 图 D: 可学习臂原始参数
ax = fig.add_subplot(gs[1, 1])
allr = np.concatenate([[SOFT[s][0], SOFT[s][1], STE[s][0], STE[s][1]] for s in seeds])
ax.axhspan(0, 0.5, color="#e8f5e9", zorder=0, label="合法区间 [0, 0.5]")
for name, D, c, m in [("soft", SOFT, "#6a1b9a", "o"), ("binary-STE", STE, "#c62828", "^")]:
    xs = [D[s][0] for s in seeds]; ys = [D[s][1] for s in seeds]
    ax.scatter(xs, ys, s=80, c=c, marker=m, ec="white", lw=0.8,
               label=f"{name} (θ₁, θ₂)", zorder=3)
ax.axhline(0, c="k", lw=0.8); ax.axvline(0, c="k", lw=0.8)
ax.set_xlabel("θ₁ 原始参数值"); ax.set_ylabel("θ₂ 原始参数值")
ax.set_title("D. 可学习臂：θ 跑出合法区间\n5/7 种子的 θ₂ 卡在 ≈ −4.1", fontsize=10, loc="left")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.25)

# --- 图 E: soft 臂的 normalized-energy cutoff
ax = fig.add_subplot(gs[1, 2])
w = 0.36
xs = np.arange(len(seeds))
ax.bar(xs - w/2, [soft_eff(SOFT[s][0]) for s in seeds], w, label="soft θ₁", color="#ce93d8")
ax.bar(xs + w/2, [soft_eff(SOFT[s][1]) for s in seeds], w, label="soft θ₂", color="#6a1b9a")
ax.axhline(0.25, c="#2e7d32", lw=2, ls="--", label="固定 0.25")
ax.set_xticks(xs); ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
ax.set_ylabel("归一化能量阈值 0.5·sigmoid(θ)")
ax.set_title("E. soft 臂阈值映射 = 0.5·sigmoid\n不是丢弃比例", fontsize=10, loc="left")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25, axis="y")

fig.suptitle("MetaIndux-TS 复现：频率阈值 θ 的实测分布（从 checkpoints 读出）",
             fontsize=14, fontweight="bold", y=0.975)
out_dir = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)
figure_path = out_dir / "theta_parameter_scatter.png"
report_path = out_dir / "theta_parameter_report.md"
fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

txt = "\n".join(L)
print(txt)
with report_path.open("w", encoding="utf-8") as f:
    f.write("# θ 实测数据的统计分析\n\n```\n" + txt + "\n```\n")
print(f"\n[图已写出] {figure_path}")
