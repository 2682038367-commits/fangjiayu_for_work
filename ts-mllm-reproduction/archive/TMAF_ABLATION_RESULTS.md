# FD001 TMAF token 消融结果

本次按用户指定的五组配置进行对照。完整 TMAF 与纯时序使用同一 split/seed 的
现有结果；仅视觉、仅文本、打乱 token 三组均从同一纯时序 checkpoint 初始化，
分别重新训练 30 epochs，按验证集 RMSE 保存最优模型。新训练全部使用 RTX 4060 Ti。

## 对照设置

- `full`：保留全部 32 个 Qwen 输出 token；
- `visual_only`：attention mask 只开放第 0 个视觉前缀输出；
- `text_only`：attention mask 只开放第 1–31 个文本输出；
- `shuffled`：在每个 split 内独立跨样本置换完整 token 序列，保留原窗口和标签；
- `temporal`：不使用 Qwen 的纯时序 Patch Transformer。

打乱时采用 shuffle seed=2026 的无固定点置换，确保没有任何样本拿到自己的 token。
置换只在各自 split 内进行，不混合 train/val/test，训练和测试都使用打乱配对。

所有 TMAF 变体保持 batch size=128、lr=0.002、epochs=30、seed=42、split seed=42。
这不是只在推理时临时遮挡，而是分别训练的 token 可用性消融。

## 结果

主指标使用裁剪到 `[0,125]` 后的测试预测。

| 配置 | 最优 epoch | 验证 RMSE | 测试 RMSE | 测试 MAE | 测试 Score |
|---|---:|---:|---:|---:|---:|
| 纯时序 | 24 | 12.3749 | **13.0132** | **9.3954** | **336.51** |
| 完整 TMAF | 7 | 12.4755 | 13.3737 | 9.6078 | 385.74 |
| 仅视觉 token | 21 | 12.3347 | 13.4786 | 9.4697 | 383.20 |
| 仅文本 token | 24 | 12.3870 | 13.9067 | 10.0935 | 423.46 |
| 打乱 Qwen token | 10 | **12.0129** | 13.2447 | 9.4990 | 368.68 |

每组测试均输出 100 条有限预测，未退化为常数。三组新训练各耗时约 93 秒。

## 解释与限制

1. 本次单 seed 下，纯时序仍为最佳测试模型。当前 TMAF 没有表现出稳定增益。
2. 完整 TMAF 的 RMSE 优于仅视觉和仅文本，但打乱配对却比完整配对略好。因此
   不能据此宣称模型已经有效利用样本级跨模态对应关系；随机训练差异、融合学习率
   或全局统计先验都有可能影响结果，需要多 seed 验证。
3. **本次是 Qwen 输出 token 消融，不是严格的输入模态移除。** Qwen 为因果模型，
   第 0 个视觉 token 不会看到后续文本，但第 1–31 个文本输出在缓存生成时已经
   能注意视觉前缀。因此“仅文本 token”仍可能保留视觉信息。若复现论文的真正
   no-visual 模态消融，必须在 Qwen 输入端去掉视觉前缀并重新生成文本缓存。
4. “仅视觉 token”也不等同于重新训练无文本 SVLMA：当前 projector 曾使用动态
   prompt embedding 对齐。严格 no-text 版本需要独立定义其 projector 训练目标。

下一步优先做：较小时间分支学习率/分阶段解冻、多 seed 稳定性实验，以及真正的
输入端无视觉缓存。不要将本表直接表述为论文完整模态贡献已被复现。

## 文件与复跑

```bash
PYTHONPATH=src ../.venv/bin/python scripts/run_tmaf_ablations.py
```

单独运行某个变体：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_tmaf.py \
  --token-mode text_only \
  --output-dir artifacts/tmaf_ablation/text_only/FD001/seed42
```

三组新产物位于 `artifacts/tmaf_ablation/{visual_only,text_only,shuffled}/FD001/seed42/`，
每组均保存 `best.pt`、`history.json`、`result.json`、预测 CSV 和 SVG 曲线。
