"""
扫描 results/ 下的所有 csv / json，打印「臂 × seed × 指标」统一表
==========================================================================
用法（在项目根目录，即含 .venv 和 results/ 的那一层）：
    .venv/bin/python collect_seed_metrics.py
    .venv/bin/python collect_seed_metrics.py results runs     # 指定要扫的目录

输出可直接整段贴回给我，用来画 θ–性能散点。
"""
import csv
import glob
import json
import os
import re
import sys

# ---------------------------------------------------------------- 列名识别
SEED_KEYS = ("seed", "random_seed", "rand_seed", "s")
RMSE_KEYS = ("rmse", "root_mean", "test_rmse", "score")
DS_KEYS = ("ds", "dss", "ds_score", "diff_std", "diversity", "metric_ds")
MAE_KEYS = ("mae",)
THETA_KEYS = ("theta", "threshold", "quantile")
NAME_KEYS = ("arm", "run", "name", "tag", "experiment", "exp", "variant", "dataset", "data")

# 强制「精确匹配优先」：避免 'seed' 命中 'seed_loss'
def pick(d, keys, exact_first=True):
    low = {k.lower(): k for k in d}
    if exact_first:
        for k in keys:
            if k in low:
                return low[k]
    for k in keys:
        for lk, ok in low.items():
            if k in lk:
                return ok
    return None


def as_num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", v)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def arm_from_path(path):
    """从文件路径猜臂名：取 results/ 之后的第一层目录名，否则取文件名主干"""
    p = path.replace("\\", "/")
    m = re.search(r"results/(?:runs/)?([^/]+)/", p)
    base = os.path.splitext(os.path.basename(p))[0]
    if m and m.group(1) not in ("runs", "csv", "json"):
        return f"{m.group(1)}/{base}" if base not in m.group(1) else m.group(1)
    return base


def seed_from_name(name):
    m = re.search(r"seed[_\-]?(\d+)", name, re.I)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- 采集
rows = []
roots = sys.argv[1:] or ["results", "outputs", "logs"]
files = []
for r in roots:
    for ext in ("csv", "json", "jsonl"):
        files += glob.glob(os.path.join(r, "**", f"*.{ext}"), recursive=True)
files = sorted(set(files))

print(f"[扫描] 目录 {roots} → 命中 {len(files)} 个 csv/json 文件\n")

for f in files:
    try:
        if f.endswith(".csv"):
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                recs = [r for r in csv.DictReader(fh) if r]
        elif f.endswith(".jsonl"):
            recs = []
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        else:
            obj = json.load(open(f, encoding="utf-8", errors="replace"))
            if isinstance(obj, list):
                recs = [o for o in obj if isinstance(o, dict)]
            elif isinstance(obj, dict):
                # 单条记录，或 {seed: {...}} 形式
                recs = [obj]
                for k, v in obj.items():
                    if isinstance(v, dict) and re.fullmatch(r"\d+", str(k)):
                        recs.append({"seed": int(k), **v})
            else:
                continue
    except Exception as e:
        print(f"  [跳过] {f}: {type(e).__name__}: {e}")
        continue

    arm = arm_from_path(f)
    for r in recs:
        if not isinstance(r, dict):
            continue
        flat = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                flat[k] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float, str)):
                        flat[f"{k}.{k2}"] = v2
        sk, rk = pick(flat, SEED_KEYS), pick(flat, RMSE_KEYS)
        if rk is None:
            continue
        dk = pick(flat, DS_KEYS)
        nk = pick(flat, NAME_KEYS)
        rows.append({
            "file": f,
            "arm": str(flat.get(nk, arm)) if nk else arm,
            "seed": as_num(flat[sk]) if sk else seed_from_name(f),
            "rmse": as_num(flat[rk]),
            "ds": as_num(flat[dk]) if dk else None,
            "extra": {k: v for k, v in flat.items()
                      if any(x in k.lower() for x in THETA_KEYS)},
        })

# ---------------------------------------------------------------- 输出
if not rows:
    print("没找到任何含 RMSE 类的数值记录。请先跑：ls results/")
    sys.exit(0)

rows = [r for r in rows if r["rmse"] is not None]

# 去重：同 arm+seed 只保留一条（后出现者优先）
seen, uniq = {}, []
for r in rows:
    key = (r["arm"], r["seed"])
    if key in seen:
        uniq[seen[key]] = r
    else:
        seen[key] = len(uniq)
        uniq.append(r)
rows = uniq

print("=" * 104)
print("统一表（可直接整段贴回）：")
print("=" * 104)
print(f"{'arm':<44}| {'seed':>6} | {'RMSE':>10} | {'DS':>10} | 额外 θ 字段")
print("-" * 104)
for r in sorted(rows, key=lambda x: (str(x["arm"]), -1 if x["seed"] is None else x["seed"])):
    sd = "None" if r["seed"] is None else f"{int(r['seed'])}"
    ds = "—" if r["ds"] is None else f"{r['ds']:.4f}"
    ex = ", ".join(f"{k}={v}" for k, v in r["extra"].items()) or "—"
    print(f"{r['arm'][:43]:<44}| {sd:>6} | {r['rmse']:>10.4f} | {ds:>10} | {ex}")

print("-" * 104)
print(f"合计 {len(rows)} 条记录，来自 {len(set(r['file'] for r in rows))} 个文件\n")

# 按臂汇总
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    agg[r["arm"]].append(r["rmse"])
print("=" * 104)
print("按臂汇总（μ ± SD, n）")
print("=" * 104)
for a, vs in sorted(agg.items()):
    mu = sum(vs) / len(vs)
    sd = (sum((v - mu) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5 if len(vs) > 1 else 0.0
    print(f"{a[:50]:<52}| {mu:8.3f} ± {sd:7.4f}  (n={len(vs)})")

print("\n提示：如果没有出现含 'seed' 的行，说明 results 里的记录没写种子号——")
print("      那就需要看 results/runs/*/metrics.json 或训练脚本里的保存逻辑。")
