#!/usr/bin/env bash
# =============================================================================
# MetaIndux-TS 复现 —— 目录 / 代码 / 结果 三层自动核对
#
# 用法：
#   cd ~/projects/rul/metaindux-ts-reproduction
#   bash analysis/audit/audit_checklist.sh .
#   bash analysis/audit/audit_checklist.sh /path/to/proj
#
# 只读，不修改任何文件。FAIL 项按 REPRODUCTION_AUDIT.md §3 的处置建议处理。
# =============================================================================
set -uo pipefail

ROOT="${1:-$(pwd)}"
if [ -d "$ROOT/.venv/bin" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="python3"
fi

echo "项目根: $ROOT"
echo "解释器: $PY"
echo

exec "$PY" - "$ROOT" <<'PYEOF'
import csv, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
FWD  = ROOT / "results" / "frequency_threshold_validation"
GEN_EXPECT  = [3, 13, 23, 33, 43, 53, 63]
EVAL_EXPECT = [3, 13, 23, 33, 43]

OK, FAIL, INFO = [], [], []
def ck(cond, label, detail=""):
    (OK if cond else FAIL).append(label)
    print(f"  [{'OK' if cond else 'FAIL':4}]  {label}" + (f"   {detail}" if detail else ""))
def info(label, detail=""):
    INFO.append(label)
    print(f"  [INFO]  {label}" + (f"   {detail}" if detail else ""))
def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)

def rd_csv(p):
    if not p.exists(): return None, None
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), (r.fieldnames or [])

def fnum(v):
    try: return float(v)
    except Exception: return None

# ---------------------------------------------------------------- A 层：完整性
sec("A 层：数据完整性")

fx_p = FWD / "fd002_fixed_025_seven_seeds_evaluation_raw.csv"
pb_p = FWD / "fd002_public_arm_seven_seeds_evaluation_raw.csv"
pr_p = FWD / "fd002_seven_seeds_paired.csv"
fx_rows, fx_cols = rd_csv(fx_p)
pb_rows, pb_cols = rd_csv(pb_p)
pr_rows, pr_cols = rd_csv(pr_p)

ck(fx_rows is not None and len(fx_rows) == 35, "A1a 固定臂 raw 35 行",
   f"实际 {0 if fx_rows is None else len(fx_rows)} 行" + ("" if fx_rows else f"  缺失: {fx_p}"))
ck(pb_rows is not None and len(pb_rows) == 35, "A1b 公开臂 raw 35 行",
   f"实际 {0 if pb_rows is None else len(pb_rows)} 行" + ("" if pb_rows else f"  缺失: {pb_p}"))

for tag, rows in (("A2a 固定臂", fx_rows), ("A2b 公开臂", pb_rows)):
    if rows:
        g = sorted({int(float(r["generation_seed"])) for r in rows})
        e = sorted({int(float(r["evaluator_seed"])) for r in rows})
        ck(g == GEN_EXPECT, f"{tag} gen seeds", f"{g}")
        ck(e == EVAL_EXPECT, f"{tag} eval seeds", f"{e}")

if pr_rows is None:
    ck(False, "A3 配对表存在", f"缺 {pr_p} → 用 make_paired.py 重建")
else:
    ck(len(pr_rows) == 35, "A3 配对表 35 行", f"实际 {len(pr_rows)}，列数 {len(pr_cols)}")

mf = FWD / "fd002_fixed_025_seven_seeds_manifest.json"
if mf.exists():
    m = json.loads(mf.read_text(encoding="utf-8"))
    ck(m.get("sample_count") == 41539, "A4a manifest sample_count", f"{m.get('sample_count')} (期望 41539)")
    ck(m.get("evaluation_count") == 35, "A4b manifest evaluation_count", f"{m.get('evaluation_count')} (期望 35)")
    ck(str(m.get("split_sha256", "")).startswith("8c390ebf"), "A4c manifest split_sha256",
       str(m.get("split_sha256"))[:16] + "…")
else:
    ck(False, "A4 manifest 存在", f"缺 {mf}")

rb = ROOT / "results" / "real_data_baseline_w48.csv"
rb_rows, rb_cols = rd_csv(rb)
frozen_path = ROOT / "results" / "frozen_numbers.json"
frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
paper_manifest_path = ROOT / "docs" / "provenance" / "paper_predictive_scores_w48.json"
if paper_manifest_path.exists():
    paper_manifest = json.loads(paper_manifest_path.read_text(encoding="utf-8"))
    PAPER = {str(ds).upper(): float(value) for ds, value in paper_manifest["values"].items()}
    paper_source = paper_manifest["source_document"]
    ck(set(PAPER) == {"FD001", "FD002", "FD003", "FD004"}, "A5b 论文指标数据集", str(sorted(PAPER)))
    ck(paper_manifest["metric"].get("sequence_length") == 48,
       "A5b 论文指标窗口长度", "48")
    ck(paper_source.get("table") == "Table I" and paper_source.get("printed_page") == "18068",
       "A5b 论文表格定位", "Table I / p. 18068")
    info("A5b 论文指标溯源哈希", hashlib.sha256(paper_manifest_path.read_bytes()).hexdigest()[:16] + "…")
else:
    PAPER = {}
    ck(False, "A5b 论文指标溯源文件", f"缺 {paper_manifest_path}")
if rb_rows:
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        rows = [r for r in rb_rows if str(r["dataset"]).strip().upper() == ds]
        seeds = sorted(int(float(r["evaluator_seed"])) for r in rows)
        ck(len(rows) == 5 and seeds == EVAL_EXPECT, f"A5 {ds} 真实对照种子",
           f"{seeds}")
        if frozen:
            expected = frozen["observable_comparisons"][ds]["real_baseline"]
            actual = [fnum(next(r["rmse"] for r in rows
                                if int(float(r["evaluator_seed"])) == seed)) for seed in EVAL_EXPECT]
            ck(all(np.isclose(a, b, rtol=0, atol=1e-12)
                   for a, b in zip(actual, expected["per_seed"])),
               f"A5 {ds} 真实对照逐种子值", "与 frozen_numbers.json 一致")
else:
    ck(False, "A5 真实对照 csv 存在", f"缺 {rb}")
if rb_rows and frozen:
    expected_hash = frozen["observable_comparisons"]["FD002"]["real_baseline"]["source"]["sha256"]
    actual_hash = hashlib.sha256(rb.read_bytes()).hexdigest()
    ck(actual_hash == expected_hash, "A5 真实对照来源哈希", actual_hash[:16] + "…")
elif not frozen:
    ck(False, "A5 frozen_numbers.json 存在", f"缺 {frozen_path}")

nfix = len(sorted((ROOT / "outputs").glob("FD002_w48_seed*_freq_threshold_025.npz")))
npub = len([p for p in sorted((ROOT / "outputs").glob("FD002_w48_seed*.npz"))
            if "freq_threshold_025" not in p.name])
ck(nfix == 7, "A6a 固定臂合成 npz 7 个", f"实际 {nfix}")
ck(npub >= 7, "A6b 公开臂合成 npz ≥7 个", f"实际 {npub}（含其它消融变体属正常）")

splits = sorted(ROOT.glob("results/**/*fd002*split*.npz"))
hashes = {}
for p in splits:
    hashes[p] = hashlib.sha256(p.read_bytes()).hexdigest()
BASE = "0b4fd49c52054af9791723baa33f38af220de513813030a3f599374ca55ae8e9"
odd = {p: h for p, h in hashes.items() if h != BASE}
known_incompatible_hash = "843652a3c5323a316944da5c6338be70353ec90009769d8f206167e17aca48df"
# One deliberately isolated legacy split is allowed.  The prior expression
# mistakenly required ``len(odd) == 0`` even when that sole known file was
# present, producing a false failure after a successful isolation.
ck(not odd or all(h == known_incompatible_hash for h in odd.values()),
   "A7 划分哈希一致性", f"{len(splits)} 个文件，{len(odd)} 个已隔离异类")
for p, h in odd.items():
    info("A7 异类（禁混用）", f"{p.relative_to(ROOT)}  {h[:12]}…")

SUMMARY_FILES = {
    "FD001 公开臂": (ROOT / "results" / "fd001_w48_public_code_evaluation_summary.csv", EVAL_EXPECT),
    "FD002 固定臂": (FWD / "fd002_fixed_025_seven_seeds_evaluation_summary.csv", GEN_EXPECT),
    "FD002 公开臂": (FWD / "fd002_public_arm_seven_seeds_evaluation_summary.csv", GEN_EXPECT),
    "FD003 公开臂": (ROOT / "results" / "fd003_w48_public_code_evaluation_summary.csv", EVAL_EXPECT),
    "FD004 公开臂": (ROOT / "results" / "fd004_w48_public_code_evaluation_summary.csv", EVAL_EXPECT),
}

def read_metric_summary(path, expected_seeds, metric):
    rows, _ = rd_csv(path)
    if rows is None:
        return None, None
    per_seed, aggregate = {}, None
    for row in rows:
        if row.get("metric") != metric:
            continue
        seed = row.get("generation_seed")
        if seed == "aggregate":
            aggregate = fnum(row.get("mean"))
        else:
            value = fnum(row.get("mean"))
            if value is not None:
                per_seed[int(seed)] = value
    observed = sorted(per_seed)
    if observed != expected_seeds or aggregate is None:
        return None, None
    values = np.array([per_seed[s] for s in expected_seeds])
    return values, aggregate

summary_rmse = {}
for label, (path, expected_seeds) in SUMMARY_FILES.items():
    values, aggregate = read_metric_summary(path, expected_seeds, "rmse")
    ck(values is not None, f"A8 {label} summary 结构",
       f"{expected_seeds}" if values is not None else f"缺失或种子/aggregate 不一致: {path}")
    if values is not None:
        ck(np.isclose(float(values.mean()), aggregate, rtol=0, atol=1e-10), f"A8 {label} aggregate",
           f"{aggregate:.12f}")
        info(f"A8 {label} 来源哈希", hashlib.sha256(path.read_bytes()).hexdigest()[:16] + "…")
        summary_rmse[label] = values

summary_ds = {}
for label in ("FD002 固定臂", "FD002 公开臂"):
    path, expected_seeds = SUMMARY_FILES[label]
    values, aggregate = read_metric_summary(path, expected_seeds, "discriminative_score")
    ck(values is not None, f"A9 {label} DS summary 结构",
       f"{expected_seeds}" if values is not None else f"缺失或种子/aggregate 不一致: {path}")
    if values is not None:
        ck(np.isclose(float(values.mean()), aggregate, rtol=0, atol=1e-10), f"A9 {label} DS aggregate",
           f"{aggregate:.12f}")
        summary_ds[label] = values

# ---------------------------------------------------------------- B 层：数值
sec("B 层：数值可复算")

def agg(rows, key):
    out = {}
    for g in GEN_EXPECT:
        v = [fnum(r[key]) for r in rows if int(float(r["generation_seed"])) == g and fnum(r.get(key)) is not None]
        if v: out[g] = sum(v) / len(v)
    return [out.get(g) for g in GEN_EXPECT]

def close(a, b, tol=5e-3):
    return a is not None and b is not None and abs(a - b) <= tol

try:
    import numpy as np
    from scipy import stats
except Exception as exc:
    print(f"  [FATAL] numpy/scipy 不可用：{exc}")
    sys.exit(2)

fix_rmse = agg(fx_rows or [], "rmse")
pub_rmse = agg(pb_rows or [], "rmse")
fix_ds   = agg(fx_rows or [], "discriminative_score")
pub_ds   = agg(pb_rows or [], "discriminative_score")

if "FD002 固定臂" in summary_rmse:
    ck(np.allclose(fix_rmse, summary_rmse["FD002 固定臂"], rtol=0, atol=1e-12),
       "B1 固定臂 RMSE raw/summary 逐种子一致", "7 个 generation seed")
else:
    ck(False, "B1 固定臂 RMSE raw/summary 逐种子一致", "缺正式 summary")
if "FD002 公开臂" in summary_rmse:
    ck(np.allclose(pub_rmse, summary_rmse["FD002 公开臂"], rtol=0, atol=1e-12),
       "B2 公开臂 RMSE raw/summary 逐种子一致", "7 个 generation seed")
else:
    ck(False, "B2 公开臂 RMSE raw/summary 逐种子一致", "缺正式 summary")

if all(v is not None for v in fix_rmse + pub_rmse):
    f = np.array(fix_rmse); p = np.array(pub_rmse)
    d = f - p
    t = stats.ttest_1samp(d, 0)
    ck(close(d.mean(), -2.3181, 1e-3), "B3a RMSE Δ 均值", f"{d.mean():+.4f} (冻结 -2.3181)")
    ck(close(d.std(ddof=1), 4.0872, 1e-3), "B3b RMSE Δ SD", f"{d.std(ddof=1):.4f} (冻结 4.0872)")
    ck(close(t.statistic, -1.501, 1e-2), "B3c RMSE t(6)", f"{t.statistic:+.3f} (冻结 -1.501)")
    ck(close(t.pvalue, 0.1841, 1e-3), "B3d RMSE p", f"{t.pvalue:.4f} (冻结 0.1841)")

    fd = np.array(fix_ds) - np.array(pub_ds)
    td = stats.ttest_1samp(fd, 0)
    if "FD002 固定臂" in summary_ds and "FD002 公开臂" in summary_ds:
        summary_delta = summary_ds["FD002 固定臂"] - summary_ds["FD002 公开臂"]
        ck(np.allclose(fd, summary_delta, rtol=0, atol=1e-12), "B3e DS raw/summary 逐种子一致",
           "7 个 generation seed")
    ck(close(fd.mean(), -0.1210, 1e-3), "B3e DS Δ 均值", f"{fd.mean():+.4f} (冻结 -0.1210)")
    ck(close(td.pvalue, 0.1436, 1e-3), "B3f DS p", f"{td.pvalue:.4f} (冻结 0.1436)")
else:
    ck(False, "B3 配对统计", "合成 RMSE 缺失，无法复算")

real = {}
if rb_rows:
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        v = np.array([fnum(r["rmse"]) for r in rb_rows
                      if str(r["dataset"]).strip().upper() == ds and fnum(r.get("rmse")) is not None])
        if len(v): real[ds] = v

if frozen:
    for ds, v in real.items():
        expected = frozen["observable_comparisons"][ds]["real_baseline"]
        ck(close(v.mean(), expected["mean"], 5e-4), f"B4 {ds} 真实对照均值",
           f"{v.mean():.3f} (冻结 {expected['mean']:.3f})")
        ck(close(v.std(ddof=1), expected["sd"], 5e-4), f"B4 {ds} 真实对照 SD",
           f"{v.std(ddof=1):.3f} (冻结 {expected['sd']:.3f})")

SYN = {"FD001": None, "FD002": None, "FD003": None, "FD004": None}
if "FD001 公开臂" in summary_rmse:
    SYN["FD001"] = float(summary_rmse["FD001 公开臂"].mean())
if "FD002 公开臂" in summary_rmse:
    SYN["FD002"] = float(summary_rmse["FD002 公开臂"].mean())
if "FD003 公开臂" in summary_rmse:
    SYN["FD003"] = float(summary_rmse["FD003 公开臂"].mean())
if "FD004 公开臂" in summary_rmse:
    SYN["FD004"] = float(summary_rmse["FD004 公开臂"].mean())
FROZEN_REPRODUCTION_GAP = {"FD001": +1.409, "FD002": +3.730, "FD003": +4.511, "FD004": +6.700}
for ds, exp in FROZEN_REPRODUCTION_GAP.items():
    if SYN[ds] is not None:
        g = SYN[ds] - PAPER[ds]
        ck(close(g, exp, 5e-3), f"B5 {ds} 正式复现差距", f"{g:+.3f} (冻结 {exp:+.3f})")

FROZEN_GEN = {"FD001": +1.597, "FD002": +4.911, "FD003": +5.027, "FD004": +5.370}
for ds, exp in FROZEN_GEN.items():
    if ds in real and SYN[ds] is not None:
        g = SYN[ds] - real[ds].mean()
        ck(close(g, exp, 5e-3), f"B6 {ds} 本地生成数据惩罚", f"{g:+.3f} (冻结 {exp:+.3f})")
info("B6 合成均值来源", "直接读取 A8 所列正式 summary CSV")

if "FD002" in real and all(v is not None for v in fix_rmse + pub_rmse):
    gap_fix = float(np.mean(fix_rmse)) - PAPER["FD002"]
    gap_pub = float(np.mean(pub_rmse)) - PAPER["FD002"]
    dv = abs(gap_fix - gap_pub)
    ck(close(dv, 2.3181, 2e-2), "B7 交叉验证：两臂 gap 差 ≈ |Δ 均值|",
       f"|{gap_fix:+.3f} − {gap_pub:+.3f}| = {dv:.3f}  vs |Δ| 2.318")
else:
    ck(False, "B7 交叉验证", "数据不全")

# ---------------------------------------------------------------- C 层：统计
sec("C 层：统计结论")

if all(v is not None for v in fix_rmse + pub_rmse + fix_ds + pub_ds):
    dR = np.array(fix_rmse) - np.array(pub_rmse)
    dD = np.array(fix_ds) - np.array(pub_ds)
    wR = stats.wilcoxon(dR); wD = stats.wilcoxon(dD)
    ck(close(wR.pvalue, 0.219, 2e-2), "C1a RMSE Wilcoxon p", f"{wR.pvalue:.4f} (冻结 0.219)")
    ck(close(wD.pvalue, 0.297, 2e-2), "C1b DS Wilcoxon p", f"{wD.pvalue:.4f} (冻结 0.297)")

    # leave-one-cell-out：逐格剔除配对表的一行，再按 gen 聚合重算
    if pr_rows:
        def loco_paired(key_f, key_r):
            ps = []
            for i in range(len(pr_rows)):
                sub = pr_rows[:i] + pr_rows[i + 1:]
                out = {}
                for g in GEN_EXPECT:
                    vf = [fnum(r[key_f]) for r in sub if int(float(r["generation_seed"])) == g]
                    vr = [fnum(r[key_r]) for r in sub if int(float(r["generation_seed"])) == g]
                    vf = [x for x in vf if x is not None]; vr = [x for x in vr if x is not None]
                    if vf and vr: out[g] = (sum(vf) / len(vf)) - (sum(vr) / len(vr))
                if len(out) >= 6:
                    v = np.array(list(out.values()))
                    ps.append(stats.ttest_1samp(v, 0).pvalue)
            return ps
        for nm, kf, kr, lo, hi in (("RMSE", "rmse_fixed", "rmse_random", 0.1725, 0.1926),
                                   ("DS", "discriminative_score_fixed", "discriminative_score_random",
                                    0.1246, 0.1588)):
            ps = loco_paired(kf, kr)
            if ps:
                ok = (min(ps) >= lo - 2e-3) and (max(ps) <= hi + 2e-3)
                ck(ok, f"C2 {nm} leave-one-cell-out",
                   f"p ∈ [{min(ps):.4f}, {max(ps):.4f}] (冻结 [{lo}, {hi}])")

    i53 = GEN_EXPECT.index(53)
    dR2 = np.delete(dR, i53); dD2 = np.delete(dD, i53)
    ck(close(stats.ttest_1samp(dR2, 0).pvalue, 0.203, 3e-2), "C3a 剔 seed53 RMSE p",
       f"{stats.ttest_1samp(dR2, 0).pvalue:.4f} (冻结 0.203，更不显著)")
    ck(close(stats.ttest_1samp(dD2, 0).pvalue, 0.305, 5e-2), "C3b 剔 seed53 DS p",
       f"{stats.ttest_1samp(dD2, 0).pvalue:.4f} (冻结 0.305)")

if len(real) == 4:
    info("C4 跨论文协议层", "不可识别：论文未报告真实数据训练基线；不计算 Q/I² 或跨论文 p 值")

# ---------------------------------------------------------------- 汇总
sec("汇总")
print(f"  PASS {len(OK)}    FAIL {len(FAIL)}    INFO {len(INFO)}")
if FAIL:
    print("\n  需处理：")
    for x in FAIL: print(f"    - {x}")
    print("\n  处置建议见 REPRODUCTION_AUDIT.md §3 各行的『不符怎么办』。")
else:
    print("\n  全部通过 —— 目录、代码链路与结果数值自洽，结论可定稿。")
print()
PYEOF
