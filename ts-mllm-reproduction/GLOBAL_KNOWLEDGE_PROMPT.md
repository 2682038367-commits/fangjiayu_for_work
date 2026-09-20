# 全局 Min-Max + 工况知识文本

本页描述历史摘要式 v1。当前执行入口已升级为
[有序工况序列 v2](ORDERED_OPERATING_PROMPT.md)，新旧产物分目录保存。

当前主流程恢复 `data_window40_stride50/split_seed42` 的逐传感器全局 Min-Max，
不使用 `data_condition_minmax` 或其时间权重、MAE、对齐与缓存。原工况归一化实验保留为额外实验。
窗口40、样本步长50、Patch步长1及既有论文明确参数不变。

## 知识来源和假设边界

[NASA C-MAPSS 数据说明](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data)
明确说明三个运行设置显著影响发动机表现，发动机间存在初始磨损及制造差异，
数据包含传感器噪声，故障随运行逐渐发展。这些事实写入 Domain。
“不能仅凭工况相关的传感器跳变判定退化，应结合持续的多传感器变化”是我们的推理指引，
不是论文公开的原始 prompt，也不是 NASA 给出的定量诊断规则。

不虚构每个工况下某传感器应升降多少，不将原始三个 setting 直接命名为未经核实的物理单位。
完整 prompt 未公开，因此本版本 `operating_context_v1` 是复现假设，不能称为完全忠实复现。
仅训练集拟合归一化参数、目标 `RUL/125` 等既有未明确假设继续保留；标签上限仍是125。

## 每窗口实际输入

- Dataset description：子集、单/多工况、40周期、14个全局归一化传感器。
- Domain：资料支持的通用知识及单独标明的推理指引。
- Observed context：当前窗口三个 setting 的组合占比、组合切换次数及末时刻原始 setting。
- Sensor summary：离散程度、相邻时刻波动及最大首末变化。
- Instruction：结合工况和传感器证据估计剩余周期。

setting1/2/3分别四舍五入至0/2/0位以合并微小抖动；这只是文本摘要假设，
没有聚类拟合、没有修改传感器值。短测试轨迹只统计真实观测行，不把补齐行算作观测。
不读取未来轨迹、真实RUL或发动机ID作为 prompt 内容。
原始数据文件、窗口末时刻、知识文字及模板哈希均核验并记录，避免混用旧对齐或缓存。
本次预检 FD002/FD004 六个 split，prompt 长度327–351 tokens，均低于512。

## 执行

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_global_knowledge_tmaf.py
```

默认处理 FD002、FD004。需要四集时加 `--datasets FD001 FD002 FD003 FD004`；
只检查命令而不运行可加 `--dry-run`。
复用来源匹配的全局归一化时间权重和 MAE（缺少匹配 MAE 时先预训练），
再重新对齐、重建频谱/Qwen缓存、训练完整TMAF和独立核验。
训练使用 GPU，TMAF 保持30轮、batch128、Adam学习率0.002、不冻结时间分支。
新输出独立保存在 `artifacts/global_knowledge_v1/<FDxxx>/seed42/`。
本次只修改代码和检查 prompt/测试，没有启动训练，也没有新性能结论。
