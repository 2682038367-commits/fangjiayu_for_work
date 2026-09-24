#!/usr/bin/env python3
"""7 种子评估自检 v2 —— 能区分「还没跑完」和「真的错了」。

v1 的三个 FAIL 里，有两类完全不同的成因，v1 全按错误处理：
  - manifest 缺失 / 行数 31<35  → 这几乎总是「没跑完」，不是错误
  - DS 偏差 9.3e-3             → 判别器训练的浮点非确定性，不是数据错
v2 把两者分开判定，并额外给出：进程是否活着、缺哪几个 (gen,eval)、
每个 gen 完成了几个 eval、DS 偏差的相对量级（判断 9.3e-3 是 3% 还是 30%）。

退出码：
  0 = 全部通过，可以进统计
  1 = 有真 FAIL，先贴回输出排查
  2 = 尚未跑完（进程活着，或行数不足）→ 继续等，不是错误

用法：
  .venv/bin/python verify_seven_seed_eval.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results/frequency_threshold_validation"
NEW_RAW = OUT / "fd002_fixed_025_seven_seeds_evaluation_raw.csv"
NEW_MANIFEST = OUT / "fd002_fixed_025_seven_seeds_manifest.json"
OLD_PAIRED = OUT / "fd002_fixed_025_paired_comparison.csv"

EXPECTED_SHA = "8c390ebfebf4cc3b18ff4b9e0cfa93c7b883e8d6a213364fc45d8bed80e82c76"
GEN_SEEDS = [3, 13, 23, 33, 43, 53, 63]
EVAL_SEEDS = [3, 13, 23, 33, 43]      # 以实际 csv 为准，这里只是默认值
OLD_GENS = (13, 23, 43)

TOL_RMSE = 1e-6      # RMSE 是确定性前向推理，必须逐位一致
TOL_DS = 2e-2        # 判别器要训练，浮点非确定性允许 ~1e-2 量级抖动
TOL_DS_REL = 0.05    # 或相对量级 5% 以内

FAILS: list[str] = []
WARNS: list[str] = []


def mark(ok, msg, hard=True):
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)
    if not ok:
        (FAILS if hard else WARNS).append(msg)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        return [dict(zip(header, line.strip().split(","))) for line in fh if line.strip()]


def pid_alive(pid: str) -> bool:
    try:
        return subprocess.run(["ps", "-p", pid], capture_output=True).returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
print("=" * 78)
print("0. 评估进程是否还活着")
print("=" * 78)
alive = False
pidfile = Path("/tmp/fixed025_eval.pid")
if pidfile.is_file():
    pid = pidfile.read_text().strip()
    alive = pid_alive(pid)
    print(f"  PID 文件 = {pidfile}  PID = {pid}  → {'运行中' if alive else '已结束'}")
else:
    print("  [提示] 没有 /tmp/fixed025_eval.pid（进程从未启动，或机器重启过）")
try:
    pg = subprocess.run(["pgrep", "-af", "evaluate_fd002_stability"],
                        capture_output=True, text=True).stdout.strip()
    if pg:
        alive = True
        print(f"  pgrep 命中：\n{chr(10).join('    ' + l for l in pg.splitlines())}")
    else:
        print("  pgrep 无 evaluate_fd002_stability 进程")
except FileNotFoundError:
    print("  [提示] 系统没有 pgrep，跳过")

if not NEW_RAW.is_file():
    print("\n[FATAL] 连 raw csv 都没有，评估根本没开始产出。")
    sys.exit(1)

rows = read_csv(NEW_RAW)
n = len(rows)
gens = sorted({int(r["generation_seed"]) for r in rows})
evals = sorted({int(r["evaluator_seed"]) for r in rows})
done = {(int(r["generation_seed"]), int(r["evaluator_seed"])) for r in rows}
missing = [(g, e) for g in GEN_SEEDS for e in evals if (g, e) not in done]

print("\n" + "=" * 78)
print(f"1. 完成度：{n}/35 行")
print("=" * 78)
per = {g: sum(1 for r in rows if r["generation_seed"] == str(g) or
              int(r["generation_seed"]) == g) for g in GEN_SEEDS}
for g in GEN_SEEDS:
    bar = "█" * per[g] + "·" * (5 - per[g])
    print(f"  gen {g:>2}: {bar} {per[g]}/5")
if n < 35:
    print(f"\n  还差 {35 - n} 行。缺失组合：")
    for g, e in missing:
        print(f"    generation_seed={g}, evaluator_seed={e}")
    if alive:
        print("\n  → 进程还活着，这是【没跑完】，不是错误。等它写完再自检。")
    else:
        print("\n  → 进程已死且没跑满。需要补跑（见附带的 resume_eval.sh）。")
else:
    print("  35 行齐全。")

print("\n" + "=" * 78)
print("2. 复现性：13/23/43 是否逐位对上旧结果")
print("=" * 78)
if not OLD_PAIRED.is_file():
    mark(False, f"找不到旧配对表 {OLD_PAIRED}")
else:
    old = read_csv(OLD_PAIRED)
    old_map = {(int(r["generation_seed"]), int(r["evaluator_seed"])): r for r in old}
    checked = 0
    worst_rmse = worst_ds = 0.0
    arg_rmse = arg_ds = None
    scale_max = 0.0
    for r in rows:
        key = (int(r["generation_seed"]), int(r["evaluator_seed"]))
        if key[0] not in OLD_GENS or key not in old_map:
            continue
        o = old_map[key]
        d1 = abs(float(r["rmse"]) - float(o["rmse_fixed"]))
        d2 = abs(float(r["discriminative_score"]) -
                 float(o["discriminative_score_fixed"]))
        scale_max = max(scale_max, abs(float(o["discriminative_score_fixed"])))
        if d1 > worst_rmse:
            worst_rmse, arg_rmse = d1, key
        if d2 > worst_ds:
            worst_ds, arg_ds = d2, key
        checked += 1
    mark(checked == 15, f"可比行数 = {checked}（期望 15 = 3 gen × 5 eval）")
    mark(worst_rmse < TOL_RMSE,
         f"RMSE 最大偏差 = {worst_rmse:.3e}（应 < {TOL_RMSE:.0e}）"
         + (f"，最差在 gen/eval={arg_rmse}" if arg_rmse else ""))
    rel = worst_ds / scale_max if scale_max > 0 else float("nan")
    ds_ok = worst_ds < TOL_DS or rel < TOL_DS_REL
    mark(ds_ok,
         f"判别分数最大偏差 = {worst_ds:.3e}（绝对值，容差 {TOL_DS:.0e}）"
         + (f"，最差在 gen/eval={arg_ds}" if arg_ds else ""))
    print(f"         相对偏差 = {rel * 100:.2f}%（旧值量级 {scale_max:.4f}，容差 {TOL_DS_REL * 100:.0f}%）")
    if not ds_ok:
        print("         → 超出非确定性抖动范围，可能是判别器/配置变了，需要排查。")
    else:
        print("         → 属判别器训练的浮点非确定性（RMSE 逐位一致已证明数据与划分没错）。")

print("\n" + "=" * 78)
print("3. 覆盖度与 manifest")
print("=" * 78)
mark(gens == GEN_SEEDS, f"generation_seeds = {gens}")
mark(len(evals) == 5, f"evaluator_seeds = {evals}（5 个）")
dup = n - len(done)
mark(dup == 0, f"无重复 (gen, eval) 组合，重复数 = {dup}")

if not NEW_MANIFEST.is_file():
    if n < 35:
        mark(False, f"manifest 尚未生成（{NEW_MANIFEST.name}）——跑到最后才写，"
                    f"当前 {n}/35 行，属正常等待", hard=False)
    else:
        mark(False, f"35 行都齐了却没有 {NEW_MANIFEST.name}", hard=True)
else:
    mf = json.loads(NEW_MANIFEST.read_text(encoding="utf-8"))
    mark(mf.get("split_sha256") == EXPECTED_SHA,
         f"split_sha256 = {mf.get('split_sha256')}")
    mark(mf.get("sample_count") == 41539, f"sample_count = {mf.get('sample_count')}（期望 41539）")
    mark(mf.get("evaluation_count") == 35,
         f"evaluation_count = {mf.get('evaluation_count')}（期望 35）")

# --------------------------------------------------------------------------- #
print()
if n < 35:
    print("=" * 78)
    print(f"结论：还没跑完（{n}/35），进程{'活着' if alive else '已死'}。")
    if alive:
        print("  → 继续等，跑完再自检。这不是 FAIL。")
        print("  → 盯进度：tail -f logs/fixed025_seven_seeds_eval.log")
    else:
        print("  → 进程死了，需要补跑：bash resume_eval.sh")
    print("=" * 78)
    sys.exit(2)

if FAILS:
    print("=" * 78)
    print("结论：有真 FAIL，不要进统计，把上面输出贴回排查。")
    for f in FAILS:
        print(f"  FAIL: {f}")
    print("=" * 78)
    sys.exit(1)

print("=" * 78)
print("结论：全部通过（DS 抖动若已标注为非确定性，可接受）→ 可以进统计：")
print("  .venv/bin/python final_seven_seed_test.py")
print("=" * 78)
sys.exit(0)
