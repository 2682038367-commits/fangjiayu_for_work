# 全局向量提取单变量对照

论文未明确如何将Qwen输出提取为全局语义向量。比较已完成的有序工况v2
有效token平均（含视觉前缀）与最后一个有效文本token隐藏状态。
后者可以通过因果注意力访问前面的视觉前缀及文本，但不能保证编码质量或性能更好。
两种提取方式都标为论文未明确的复现假设。

复用同一Qwen缓存、数据划分、时间初始化及融合初始化；保持global_broadcast、
RUL/125、30轮、batch128、Adam学习率0.002、零冻结、seed42及所有模型参数不变。
不重训MAE/对齐，不重新运行Qwen。已有mean结果保留，不重复训练。
最后有效文本位置根据mask选取，不误取padding或视觉前缀；广播后的注意力仍然均匀。

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_global_pooling_comparison.py
```

默认FD002、FD004，训练及独立回放核验要求GPU。
新输出：`artifacts/global_knowledge_sequence_v2/<FDxxx>/seed42/tmaf_last_text/`。
脚本核验源文件哈希、同初始权重及相同训练参数，打印均值/末token的验证与测试指标。
按验证RMSE选择方案，不按测试集调优；单seed结果不能宣称稳定改善。
添加`--dry-run`可仅显示训练命令。本次仅修改代码及验证，未启动训练。
