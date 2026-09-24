#!/usr/bin/env bash
cd "$HOME/projects/rul/metaindux-ts-reproduction" || { echo "[FATAL] 路径不对"; exit 1; }

echo "=== 1. 进程状态 ==="
if ps -p "$(cat /tmp/fixed025.pid 2>/dev/null)" >/dev/null 2>&1; then
  echo "还在跑"; ps -o etime= -p "$(cat /tmp/fixed025.pid)" | sed 's/^/  已运行 /'
else
  echo "已结束"
fi

echo
echo "=== 2. ckpt（应有 7 个）==="
find . -name "*_freq_threshold_025.pth" -printf "%f   %TY-%Tm-%Td %TH:%TM\n" | sort
echo "  计数: $(find . -name "*_freq_threshold_025.pth" | wc -l)"

echo
echo "=== 3. 合成数据 npz（应有 7 个，评估要读它）==="
find . -name "*_freq_threshold_025.npz" -printf "%f   %TY-%Tm-%Td %TH:%TM\n" | sort
echo "  计数: $(find . -name "*_freq_threshold_025.npz" | wc -l)"

echo
echo "=== 4. θ 校验（关键）==="
.venv/bin/python backfill_fixed_arm.py --verify 2>&1 | tail -20
