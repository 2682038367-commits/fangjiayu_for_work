#!/usr/bin/env bash
# FD002 固定 θ=0.25 臂：7 生成种子 × 5 评估种子的完整评估
#
#   前置：4 个新种子 (3/33/53/63) 训练已完成，7 个 npz 齐全
#   产出：fd002_fixed_025_seven_seeds_evaluation_raw.csv（35 行）
#         + 同名 summary / manifest / paired_comparison
#   之后自动跑 verify_seven_seed_eval.py 做确定性自检
#
set -euo pipefail

PROJ="$HOME/projects/rul/metaindux-ts-reproduction"
CFG="configs/evaluation_fd002_frequency_threshold_seven_seeds.yaml"
LOG="logs/fixed025_seven_seeds_eval.log"
SEEDS="3 13 23 33 43 53 63"

cd "$PROJ" || { echo "[FATAL] 找不到项目目录: $PROJ"; exit 1; }
[ -x .venv/bin/python ] || { echo "[FATAL] 找不到 .venv/bin/python"; exit 1; }
mkdir -p logs

echo "=== [0/3] 前置检查：7 个合成 npz 是否存在 ==="
MISSING=0
for s in $SEEDS; do
  NPZ="outputs/FD002_w48_seed${s}_freq_threshold_025.npz"
  if [ -f "$NPZ" ]; then
    echo "  [OK]   $NPZ"
  else
    echo "  [缺失] $NPZ"
    MISSING=$((MISSING + 1))
  fi
done
if [ "$MISSING" != "0" ]; then
  echo
  echo "还有 $MISSING 个 npz 未生成，训练尚未完成。等 start_backfill.sh 跑完再来。"
  exit 1
fi

echo
echo "=== [1/3] 确认评估配置文件存在 ==="
[ -f "$CFG" ] || { echo "[FATAL] 缺少 $CFG，请先放入 configs/"; exit 1; }
echo "  [OK]   $CFG"
echo "  generation_seeds: $(grep -A0 'generation_seeds' "$CFG" | tr -d ' ')"

echo
echo "=== [2/3] 后台启动评估（7×5=35 次，约 45 分钟）==="
nohup .venv/bin/python scripts/evaluate_fd002_stability.py --config "$CFG" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/fixed025_eval.pid
echo "  已启动，PID=$PID，日志: $PROJ/$LOG"
echo
echo "后续："
echo "  看进度 : tail -f $LOG"
echo "  查存活 : ps -p $PID > /dev/null && echo running || echo finished"
echo "  跑完后 : .venv/bin/python verify_seven_seed_eval.py"
exit 0
