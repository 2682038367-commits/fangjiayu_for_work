# 工况归一化主流程：新文本方案对照

主推进方案按用户决定采用已验证的training-only工况Min-Max。
它是文献支持的补充预处理，不能称为TS-MLLM原文明确定义的方法。

A保留已完成的 `tmaf_condition_minmax`，原统计文本；B采用
`condition_operating_sequence_v1`：通用知识、当前窗口有序工况字典/序列、
组合占比/切换及末设置、传感器统计，并明确提示：

```text
Sensor scale: training-only per-condition Min-Max has corrected major
operating-condition scale differences; do not interpret these as raw
absolute sensor values.
```

setting舍入和短轨迹行对齐沿用有序工况v2。原全局v2模板不修改，旧产物不覆盖。
预检查FD002/FD004全部train/val/test：最长458 tokens，超512时直接报错，
不允许截断周期。51项测试通过。

## 保持不变

相同数据划分、工况归一化数据、MAE权重、时间权重、融合初始化及模型配置。
窗口40、样本stride50、Patch stride1、目标RUL/125、全局mean广播、
30epochs、batch128、Adam统一lr0.002、零冻结、seed42不变。
B重新训练对齐（包含可学习频谱融合）、生成完整Qwen缓存和训练TMAF；A无需重训。
因此是**整个新文本方案**对照，不是只增加工况描述的严格单因素实验：
通用知识与文本措辞也变化，且对齐权重重新学习。不能将差异单独归因于工况序列。
原prompt未公开，这些新增内容均为复现假设。

## 执行

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_condition_knowledge_comparison.py
```

默认GPU运行FD002、FD004；只跑一集可加 `--datasets FD002`，检查命令加 `--dry-run`。
源文件checksum、同初始化/模型/日程核验后，独立GPU回放核验模型指标与CSV，
最后打印A/B验证和测试指标。按验证RMSE判断，不用测试指标选方案。
新输出 `artifacts/condition_knowledge_sequence_v1/<FDxxx>/seed42/`；旧A保留。
本次尚未启动训练；A测试RMSE FD002=17.0862、FD004=18.9958。
