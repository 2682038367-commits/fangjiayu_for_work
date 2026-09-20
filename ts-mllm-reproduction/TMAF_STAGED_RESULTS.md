# TMAF 分阶段联合训练

**额外训练策略实验，不是论文明确方法，不能作为忠实复现主结果。**
完整核对见 [PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md)。

固定 FD001 split seed=42、完整 Qwen 缓存、纯时序预训练 seed42 checkpoint，
采用以下策略在 RTX 4060 Ti 上训练 30 epochs：

- Epoch 1–5：冻结整个时间分支，并设为 eval 关闭其 Dropout；
- Epoch 6–30：解冻时间编码器，学习率 2e-4；
- TMAF 的 Query/Key/Value、线性融合和回归头全程训练，学习率 2e-3；
- Adam/MSE、batch size=128，按验证 RMSE 保存最优模型；
- 时间 checkpoint 中未使用的原纯时序回归头始终冻结。

## Seed=42 实际结果

| 配置 | 测试 RMSE | 测试 MAE | 测试 Score |
|---|---:|---:|---:|
| 纯时序 | 13.0132 | 9.3954 | 336.51 |
| 原 TMAF：全程 lr=2e-3 | 13.3737 | 9.6078 | 385.74 |
| 分阶段 TMAF | **12.5539** | **8.8884** | **307.26** |

最优 epoch=13，验证 RMSE=12.2891；测试正好 100 条有限预测，预测标准差为
41.0335；耗时约 87.10 s。RMSE 比原 TMAF 改善约 0.8197，比纯时序改善约 0.4592。
论文 FD001 RMSE 为 12.45，当前结果仍略高，且单 seed 不能证明统计显著提升。

本次是明确记录的训练策略补充，而非论文公开的原始统一学习率设置。应继续用
相同策略运行 seed52/62，汇总均值和样本标准差，以判断跨 seed 稳定性。不要根据
测试结果选择最佳 seed。

## 复跑

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_tmaf.py \
  --temporal-learning-rate 0.0002 \
  --freeze-temporal-epochs 5
```

输出独立保存于 `artifacts/tmaf_staged/full/FD001/seed42/`，不覆盖原实验。
history 逐轮记录 temporal_frozen、有效时间学习率与融合学习率；checkpoint 和
result 记录冻结轮数及双学习率配置。冻结/解冻梯度与 Dropout 行为均有单元测试。
