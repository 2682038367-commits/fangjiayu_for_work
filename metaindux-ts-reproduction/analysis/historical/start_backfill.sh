#!/usr/bin/env bash
# 补跑 FD002 固定 θ=0.25 臂缺失的 4 个种子：3 / 33 / 53 / 63
#
#   默认只做 dry-run + 自动核对，不启动训练。
#   核对通过后执行：  bash start_backfill.sh --go
#
set -euo pipefail

PROJ="$HOME/projects/rul/metaindux-ts-reproduction"
CFG="configs/frequency_threshold_validation_fd002_remaining_seeds.yaml"
LOG="logs/fixed025_backfill.log"
WANT_SEEDS="3 33 53 63"

cd "$PROJ" || { echo "[FATAL] 找不到项目目录: $PROJ"; exit 1; }
[ -x .venv/bin/python ] || { echo "[FATAL] 找不到 .venv/bin/python"; exit 1; }
mkdir -p logs

echo "=== [0/4] 写配置文件 $CFG ==="
cat > "$CFG" <<'EOF'
experiment:
  name: metaindux_ts_fd002_fixed_frequency_threshold_validation
  artifact_tag: freq_threshold_025
  datasets: [FD002]
  window_sizes: [48]
  seeds: [3, 33, 53, 63]
model:
  name: DiffUnet_fre
  input_size: 14
  frequency_threshold: 0.25
diffusion:
  timesteps: 1000
  beta_schedule: linear
  sample_type: ddpm
training:
  epochs: 70
  optimizer: adam
  learning_rate: 0.002
  lr_schedule: warmup_cosine
  batch_size: 256
  grad_clip: 1.0
runtime:
  state: all
  device: cuda
  wandb_mode: offline
EOF
echo "已写入。"

echo
echo "=== [1/4] dry-run 预览实际命令 ==="
DRY=$(.venv/bin/python scripts/run_experiments.py --config "$CFG" --dry-run)
echo "$DRY"
echo "$DRY" > /tmp/fixed025_dryrun.txt

echo
echo "=== [2/4] 自动核对 ==="
OK=1
NUM=$(echo "$DRY" | grep -c "MainCondition.py" || true)
GOT_SEEDS=$(echo "$DRY" | grep -o -- "--seed [0-9]*" | awk '{print $2}' | sort -n | tr '\n' ' ' | sed 's/ $//')
FT=$(echo "$DRY" | grep -c -- "--frequency_threshold 0.25" || true)
ST=$(echo "$DRY" | grep -c -- "--state all" || true)
TAG=$(echo "$DRY" | grep -c "seed3_freq_threshold_025.pth" || true)

chk() { if [ "$2" = "$3" ]; then echo "  [OK]   $1 = $2"; else echo "  [FAIL] $1 = $2 （期望 $3）"; OK=0; fi }
chk "命令条数" "$NUM" "4"
chk "种子列表" "$GOT_SEEDS" "$WANT_SEEDS"
chk "固定阈值条数" "$FT" "4"
chk "state=all 条数" "$ST" "4"
chk "产物后缀 _freq_threshold_025" "$TAG" "1"

if [ "$OK" != "1" ]; then
  echo
  echo "核对未通过，已中止。请把上面输出贴回来。"
  exit 1
fi
echo "  → 全部通过。"

if [ "${1:-}" != "--go" ]; then
  echo
  echo "=== dry-run 完成，未启动训练 ==="
  echo "确认无误后执行：  bash start_backfill.sh --go"
  exit 0
fi

echo
echo "=== [3/4] 后台启动（约 2.4 小时）==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || \
  echo "  (nvidia-smi 不可用，GPU 状态未知)"
nohup .venv/bin/python scripts/run_experiments.py --config "$CFG" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/fixed025.pid
echo "  已启动，PID=$PID，日志: $PROJ/$LOG"
sleep 20
echo
echo "=== [4/4] 启动 20 秒后的日志尾部 ==="
tail -5 "$LOG" || true
echo
echo "后续操作："
echo "  看进度        : tail -f $LOG"
echo "  查是否还活着  : ps -p $PID > /dev/null && echo running || echo finished"
echo "  跑完校验 θ    : .venv/bin/python backfill_fixed_arm.py --verify   # 期望 7 行全 ✓ 已固定 0.250000"
