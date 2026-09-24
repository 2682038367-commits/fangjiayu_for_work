# 历史归因报告：已撤回

> **不要引用本文件旧结论。** 2026-09-24 核对论文指标含义后确认：论文报告的是
> “用生成数据训练”的预测分数，而未报告“用真实数据训练”的论文基线。因此不能从
> 本地真实数据基线与论文生成数据点估计的差，识别所谓协议层；同样不能计算生成端占比、
> 跨论文 t/p、Q/I² 或 FD002/FD004 的协议差异。

当前权威口径为：

- 正式复现差距 = 本地生成数据训练 RMSE − 论文生成数据训练 RMSE；
- 本地生成数据惩罚 = 本地生成数据训练 RMSE − 本地真实数据训练 RMSE。

请改用 `results/frozen_numbers.json` 中的 `observable_comparisons`，或运行
`analysis/diagnostics/protocol_gap_decomposition.py`。本文件保留路径仅为避免旧链接失效。
