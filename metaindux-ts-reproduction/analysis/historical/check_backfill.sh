#!/usr/bin/env bash
# 查 FD002 固定 θ=0.25 补跑（种子 3/33/53/63）跑完没有
# 用法：  bash check_backfill.sh
#
# 不做任何修改，只读状态。
# 项目根目录不对就改下面这行。

PROJ="$HOME/projects/rul/metaindux-ts-reproduction"
LOG="logs/fixed025_backfill.log"
PIDFILE="/tmp/fixed025.pid"
CKPT_DIR="checkpoints/FD002_w48_synthetic"   # 见下方说明，脚本会自动找
WANT_SEEDS="3 33 53 63"

cd "$PROJ" || { echo "[FATAL] 找不到项目目录: $PROJ"; exit 1; }

echo "===== 1. 进程还在不在 ====="
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if ps -p "$PID" > /dev/null 2>&1; then
    echo "  [RUNNING] PID=$PID 仍在运行"
    ps -o etime=,pcpu=,cmd= -p "$PID" | sed 's/^/    /'
  else
    echo "  [DONE]    PID=$PID 已不存在 → 训练已结束（正常结束或异常退出，看第 3 步）"
  fi
else
  echo "  (无 PID 文件，可能用了别的方式启动)"
fi

echo
echo "  —— 兜底：所有 MainCondition.py 进程 ——"
if pgrep -af "MainCondition.py" > /dev/null 2>&1; then
  pgrep -af "MainCondition.py" | sed 's/^/    /'
else
  echo "    没有 MainCondition.py 在跑"
fi

echo
echo "===== 2. 产物 .pth 到位情况 ====="
# 先在常见目录里找 *_freq_threshold_025.pth
FOUND=$(find . -name "*_freq_threshold_025.pth" -not -path "./.venv/*" 2>/dev/null | sort)
if [ -z "$FOUND" ]; then
  echo "  一个都没找到。看看 checkpoints 下都有什么："
  ls -la checkpoints 2>/dev/null | head -20
else
  echo "$FOUND" | sed 's/^/    /'
fi
echo
for s in $WANT_SEEDS; do
  HIT=$(echo "$FOUND" | grep -c "seed${s}_" || true)
  if [ "$HIT" -gt 0 ]; then
    echo "  [OK]   seed$s 的 ckpt 已生成"
  else
    echo "  [--]   seed$s 的 ckpt 还没生成"
  fi
done

echo
echo "===== 3. 日志尾部 ====="
if [ -f "$LOG" ]; then
  echo "  日志: $PROJ/$LOG   大小: $(du -h "$LOG" | awk '{print $1}')   最后修改: $(stat -c '%y' "$LOG" | cut -d. -f1)"
  echo "  —— 最后 15 行 ——"
  tail -15 "$LOG" | sed 's/^/    /'
  echo
  echo "  —— 训练进度（抓到的 epoch 行） ——"
  grep -iE "epoch|seed|Traceback|Error" "$LOG" | tail -8 | sed 's/^/    /'
  echo
  echo "  —— 异常扫描 ——"
  if grep -qiE "traceback|CUDA out of memory|RuntimeError|Killed" "$LOG"; then
    grep -inE "traceback|CUDA out of memory|RuntimeError|Killed" "$LOG" | head -5 | sed 's/^/    [!] /'
  else
    echo "    没发现 Traceback / OOM / Killed"
  fi
else
  echo "  [FATAL] 找不到日志 $LOG"
fi

echo
echo "===== 4. GPU ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /' \
  || echo "    (nvidia-smi 不可用)"

echo
echo "===== 判读 ====="
N_PTH=$(echo "$FOUND" | grep -c . || true)
if [ "$N_PTH" -ge 4 ]; then
  echo "  4 个 ckpt 都齐了 → 可以进入下一步："
  echo "    .venv/bin/python backfill_fixed_arm.py --verify"
else
  echo "  还差 $((4 - N_PTH)) 个 → 继续等。实时看进度： tail -f $LOG"
fi
