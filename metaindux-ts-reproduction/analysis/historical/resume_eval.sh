#!/usr/bin/env bash
# 七种子评估没跑满（进程死了 / 机器重启）时的补跑。
#
# 策略：备份现有 raw csv → 全量重跑 → 用 v2 自检确认 35 行。
# 全量重跑约 45 分钟；之所以不全量改增量，是因为重跑出来的 31 行必须
# 与旧 31 行逐位一致（RMSE 级别），这本身就是一次确定性复核。
#
set -euo pipefail

PROJ="$HOME/projects/rul/metaindux-ts-reproduction"
CFG="configs/evaluation_fd002_frequency_threshold_seven_seeds.yaml"
RAW="results/frequency_threshold_validation/fd002_fixed_025_seven_seeds_evaluation_raw.csv"

cd "$PROJ" || { echo "[FATAL] 找不到项目目录"; exit 1; }
[ -x .venv/bin/python ] || { echo "[FATAL] 找不到 .venv/bin/python"; exit 1; }

echo "=== [0/4] 确认没有评估进程在跑（避免抢 GPU）==="
if pgrep -af "evaluate_fd002_stability" > /dev/null 2>&1; then
  echo "  [中止] 已有评估进程："
  pgrep -af "evaluate_fd002_stability"
  echo "  先等它跑完，或用 kill 明确停掉再来。"
  exit 1
fi
echo "  [OK]   无占用进程"

echo
echo "=== [1/4] 现有成果备份 ==="
if [ -f "$RAW" ]; then
  TS=$(date +%Y%m%d_%H%M%S)
  cp -v "$RAW" "${RAW%.csv}_bak_${TS}.csv"
  echo "  已备份，行数 = $(($(wc -l < "$RAW") - 1))"
else
  echo "  [提示] 没有现成 raw csv，属首次跑"
fi

echo
echo "=== [2/4] 7 个合成 npz 检查 ==="
MISS=0
for s in 3 13 23 33 43 53 63; do
  [ -f "outputs/FD002_w48_seed${s}_freq_threshold_025.npz" ] \
    && echo "  [OK]   seed $s" || { echo "  [缺失] seed $s"; MISS=$((MISS+1)); }
done
[ "$MISS" = "0" ] || { echo "还有 $MISS 个 npz 缺失，先补训练"; exit 1; }

echo
echo "=== [3/4] 评估脚本支持的参数（确认能否只跑部分种子）==="
.venv/bin/python scripts/evaluate_fd002_stability.py --help 2>&1 | head -40 || true
echo
echo "  上面若有 --generation-seeds / --seeds 之类的参数，可以只补缺失种子；"
echo "  没有就用下面的全量重跑。"

echo
echo "=== [4/4] 全量重跑（约 45 分钟）==="
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/fixed025_seven_seeds_eval_resume_${TS}.log"
mkdir -p logs
nohup .venv/bin/python scripts/evaluate_fd002_stability.py --config "$CFG" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/fixed025_eval.pid
echo "  已启动 PID=$PID"
echo "  日志   : $PROJ/$LOG"
echo "  进度   : tail -f $LOG"
echo "  自检   : .venv/bin/python verify_seven_seed_eval.py"
exit 0
