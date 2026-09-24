#!/usr/bin/env bash
# 判定刚那次 extract 拿到的是【旧值】还是【本次新跑出来的值】
set -u
cd "${HOME}/projects/rul/metaindux-ts-reproduction" || exit 1

echo "=== 1. 训练进程是否还活着 ==="
if ps -p 1119117 >/dev/null 2>&1; then
  echo "  PID 1119117 仍在运行 → extract 拿到的一定是旧值，等它跑完再提取。"
else
  echo "  PID 1119117 已结束。"
fi

echo
echo "=== 2. 日志末尾（判断是否真的跑完 5 个种子）==="
tail -15 logs/fd002_real_baseline.log 2>/dev/null || echo "  无日志"

echo
echo "=== 3. csv 修改时间 vs 备份时间 ==="
stat -c '  %y  %n' results/real_data_baseline_w48.csv 2>/dev/null
ls -la results/real_data_baseline_w48.csv.bak.* 2>/dev/null | sed 's/^/  /'

echo
echo "=== 4. 旧备份里的 FD002 ==="
grep -h "^FD002" results/real_data_baseline_w48.csv.bak.* 2>/dev/null | sort | sed 's/^/  /' \
  || echo "  备份里没有 FD002 行"

echo
echo "=== 5. 当前 csv 里的 FD002（行数必须是 5，多出来说明是追加）==="
grep -c "^FD002" results/real_data_baseline_w48.csv | sed 's/^/  行数: /'
grep -h "^FD002" results/real_data_baseline_w48.csv | sort | sed 's/^/  /'

echo
echo "=== 6. 本次真实对照用的是哪个划分（决定同源是硬保证还是软约束）==="
grep -n "config\|yaml\|split_indices\|DEFAULT" scripts/evaluate_real_data_baseline.py | head -20 | sed 's/^/  /'

echo
echo "=== 7. 两个候选划分的 sha256 ==="
for f in results/frequency_threshold_validation/fd002_fixed_025_split.npz \
         results/frequency_threshold_validation/fd002_fixed_025_seven_seeds_split.npz; do
  [ -f "$f" ] && echo "  $(sha256sum "$f" | awk '{print $1}')  $f" || echo "  缺失 $f"
done

echo
echo "================================================================"
echo " 判读："
echo "   · 第 4 步与第 5 步数值完全一致 + 日志显示跑完 → 确定性复现，结论可信"
echo "   · 数值不同 → 以新值为准，重新跑 extract"
echo "   · 第 5 步行数 > 5 → csv 被追加了，先清理重复行再提取"
echo "================================================================"
