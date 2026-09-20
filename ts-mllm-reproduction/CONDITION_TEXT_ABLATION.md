# 工况归一化A：文本输出token消融

完整A：原统计prompt的全部有效Qwen输出token。
消融：同一缓存只开放第0个视觉前缀输出，屏蔽全部文本输出token，重新训练TMAF。
注意力mask在训练、验证及独立GPU回放均按visual_only处理，不误用full回放。

复用A的工况Min-Max数据、MAE/对齐/Qwen缓存和时间权重；同初始化、窗口40、
sample stride50、Patch stride1、mean广播、RUL/125、30epochs、batch128、
Adam lr0.002、无冻结、seed42。完整A无需重训。新结果及原结果均保留。

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_condition_text_ablation.py
```

默认GPU训练FD002、FD004，只显示命令可加`--dry-run`。
新输出在`artifacts/tmaf_condition_ablation/visual_only/<FDxxx>/seed42/`。
核验数据/缓存/时间权重checksum、同初始化及配置，然后独立GPU回放指标和CSV，
打印完整A与视觉输出消融的验证/测试指标。按验证RMSE判断，不按测试调优。

这是缓存输出级诊断，不等于完整文本输入消融：视觉编码器/桥接及可学习频谱融合
仍来自原含文本的对齐训练。移除平均池化中的数百个文本输出后，语义向量分布
也会变化，因此结果不能纯粹归因于领域知识内容。单seed不代表稳定因果增益。
本次只修改代码和验证，未启动训练。
