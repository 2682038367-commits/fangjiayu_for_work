#!/usr/bin/env bash
# RETIRED (2026-09-24): this historical script encodes an invalid
# "protocol-layer" decomposition.  Do not run it; use
# analysis/diagnostics/protocol_gap_decomposition.py for current reporting.
echo "[RETIRED] run_fd002_real_baseline_v2.sh 使用已撤回的‘协议层’归因，已禁用。"
echo "请使用 analysis/diagnostics/protocol_gap_decomposition.py；不会运行任何训练。"
exit 64

# FD002 真实数据对照（TRTR）—— v2
#
# 目的：拿到 FD002「用真实数据训练、真实数据测试」的 RMSE，
#       从而把 FD002 的总 gap 拆成两段：
#           协议层 = 真实对照 − 论文值(24.565)
#           生成端 = 合成均值 − 真实对照
#       FD002 与 FD004 同为 6 工况 × 2 故障模式，是唯一的结构同构对照，
#       用来判定 FD004 那个显著的 +1.329 是孤例还是多工况数据集的通病。
#
# 用法：
#   bash run_fd002_real_baseline_v2.sh check    # 只探查，不跑，先确认参数与格式
#   bash run_fd002_real_baseline_v2.sh run      # 真正跑（约 30 分钟）
#   bash run_fd002_real_baseline_v2.sh extract  # 只从已有 csv 提取并出结论
set -u

PROJ="${HOME}/projects/rul/metaindux-ts-reproduction"
cd "$PROJ" || { echo "[FATAL] 找不到项目目录：$PROJ"; exit 1; }

MODE="${1:-check}"
CSV="results/real_data_baseline_w48.csv"
SEEDS="3 13 23 33 43"
SCRIPT="scripts/evaluate_real_data_baseline.py"
LOG="logs/fd002_real_baseline.log"

echo "项目目录: $PROJ"
echo "模式: $MODE"
echo

# ============================================================ 探查
if [ "$MODE" = "check" ] || [ "$MODE" = "run" ]; then
  echo "=== A. 评估脚本是否存在 ==="
  if [ -f "$SCRIPT" ]; then
    echo "  ✓ $SCRIPT"
  else
    echo "  ✗ 找不到 $SCRIPT"
    echo "    列出 scripts/ 下真实对照相关的脚本："
    ls scripts/ | grep -i "real\|baseline" | sed 's/^/      /'
    exit 1
  fi

  echo
  echo "=== B. 脚本参数（确认 --datasets / --seeds 是不是这么写）==="
  .venv/bin/python "$SCRIPT" --help 2>&1 | head -40 | sed 's/^/    /'

  echo
  echo "=== C. 脚本是否支持外部划分（决定「同源」是硬保证还是软约束）==="
  if grep -nq "split" "$SCRIPT"; then
    grep -n "split" "$SCRIPT" | head -12 | sed 's/^/    /'
    echo
    echo "    ↑ 若出现 split_indices / split_npz 之类的参数，务必指向"
    echo "      results/frequency_threshold_validation/fd002_fixed_025_split.npz"
    echo "      否则真实对照与合成评估不在同一划分上，恒等式是近似成立。"
  else
    echo "    ✗ 脚本里没有任何 split 相关参数 → 划分由脚本内部按 seed 生成。"
    echo "      这意味着「同源」只能是软约束（同协议、不同具体划分）。"
    echo "      FD003/FD004 的真实对照也是这么跑的，横向可比性不受影响，"
    echo "      但报告里要写一句『真实对照与合成评估使用同一协议、独立划分』。"
  fi

  echo
  echo "=== D. 结果 csv 现状（用已有的 FD003/FD004 行验证列格式）==="
  if [ -f "$CSV" ]; then
    echo "  表头: $(head -1 "$CSV")"
    echo "  已有行："
    awk -F, 'NR>1 && ($1=="FD003"||$1=="FD004"){printf "    %s\n", $0}' "$CSV" | head -12
    N002=$(awk -F, 'NR>1 && $1=="FD002"' "$CSV" | wc -l)
    echo "  FD002 已有行数 = $N002"
    [ "$N002" -gt 0 ] && echo "    ↑ 已跑过，重跑会追加/覆盖，注意备份。"
  else
    echo "  不存在 $CSV（首次运行会新建）"
  fi

  echo
  echo "=== E. GPU 占用 ==="
  BUSY=$(pgrep -af "MainCondition.py|evaluate_fd002_stability|evaluate_real_data_baseline" | wc -l)
  if [ "$BUSY" -gt 0 ]; then
    echo "  有 $BUSY 个进程在跑："
    pgrep -af "MainCondition.py|evaluate_fd002_stability|evaluate_real_data_baseline" \
      | sed 's/^/    /' | cut -c1-110
    echo "  → 等它们结束再跑，别抢 GPU。"
  else
    echo "  无占用，可以跑。"
  fi

  if [ "$MODE" = "check" ]; then
    echo
    echo "================================================================"
    echo " 以上是探查结果。确认 B 的参数名和 D 的列格式后，再执行："
    echo "   bash run_fd002_real_baseline_v2.sh run"
    echo "================================================================"
    exit 0
  fi
fi

# ============================================================ 跑
if [ "$MODE" = "run" ]; then
  echo
  echo "=== 1. 备份现有 csv ==="
  if [ -f "$CSV" ]; then
    cp "$CSV" "${CSV}.bak.$(date +%m%d_%H%M)" && echo "  已备份 ${CSV}.bak.$(date +%m%d_%H%M)"
  else
    echo "  无现有 csv，跳过"
  fi

  echo
  echo "=== 2. 跑 FD002 真实对照（5 个评估种子，只训下游 LSTM，不生成合成数据）==="
  mkdir -p logs
  nohup .venv/bin/python "$SCRIPT" --datasets FD002 --seeds $SEEDS > "$LOG" 2>&1 &
  PID=$!
  echo "  PID=$PID   日志=$LOG   PID 文件=/tmp/fd002_real.pid"
  echo "$PID" > /tmp/fd002_real.pid
  echo
  echo "  监控："
  echo "    tail -f $LOG"
  echo "    ps -p $PID >/dev/null && echo running || echo finished"
  echo
  echo "  跑完后执行（会自动提取并出结论）："
  echo "    bash run_fd002_real_baseline_v2.sh extract"
  exit 0
fi

# ============================================================ 提取 + 出结论
if [ "$MODE" = "extract" ]; then
  echo "=== 从 csv 提取 FD002 的 RMSE（按表头定位列，不硬编码列号）==="
  [ -f "$CSV" ] || { echo "[FATAL] 没有 $CSV"; exit 1; }

  VALS=$(.venv/bin/python - "$CSV" <<'PY'
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
if not rows:
    sys.exit("[FATAL] csv 为空")
fields = list(rows[0].keys())
low = {f.lower(): f for f in fields}
print(f"# 列名: {fields}", file=sys.stderr)

def pick(*cands):
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

c_ds = pick("dataset", "data", "subset", "fd")
c_sd = pick("seed", "evaluator_seed", "eval_seed")
c_rm = pick("rmse", "rmse_mean", "test_rmse")
miss = [n for n, c in [("dataset", c_ds), ("seed", c_sd), ("rmse", c_rm)] if c is None]
if miss:
    sys.exit(f"[FATAL] 定位不到列: {miss}；实际列名见上方 # 列名")

out = []
print(f"# 使用列: dataset={c_ds}, seed={c_sd}, rmse={c_rm}", file=sys.stderr)
for r in rows:
    if str(r[c_ds]).strip().upper() == "FD002":
        out.append((int(float(r[c_sd])), float(r[c_rm])))
        print(f"#   seed {r[c_sd]:>4}  rmse {r[c_rm]}", file=sys.stderr)
if len(out) != 5:
    print(f"# [WARN] FD002 行数 = {len(out)}，期望 5", file=sys.stderr)
print(" ".join(f"{v:.6f}" for _, v in sorted(out)))
PY
)
  if [ -z "$VALS" ]; then
    echo "[失败] 没有提取到 FD002 的 RMSE，看上面的报错（多半是列名不匹配）。"
    exit 1
  fi

  echo
  echo "=== 喂给归因分解脚本 ==="
  if [ -f "protocol_gap_decomposition.py" ]; then
    .venv/bin/python protocol_gap_decomposition.py --fd002-real $VALS
  else
    echo "  本地没有 protocol_gap_decomposition.py，手动跑："
    echo "    python3 protocol_gap_decomposition.py --fd002-real $VALS"
  fi
  exit 0
fi

echo "未知模式：$MODE （可选 check / run / extract）"
exit 1
