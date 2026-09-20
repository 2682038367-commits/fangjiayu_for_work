# CNN、ViT 与 MAE 视觉编码器对比

## 对比设置

- 数据集：FD001；模型种子和发动机划分种子均为 42；
- 输入：RP/STFT/Morlet-CWT 三通道 `114×114` 图像；
- ViT/MAE：`19×19` patch，共 36 个 token，128 维、2 层、4 头；
- MAE：随机遮挡 75% patch，在 FD001 训练频谱上自监督预训练 15 epochs；
- RUL 微调：30 epochs，时序分支学习率 `2e-4`，视觉及融合层学习率 `1e-3`；
- 时序分支均从同一个 Temporal-only 最优 checkpoint 初始化；
- ViT 与 MAE 使用完全相同的固定频谱缓存，唯一区别是视觉编码器初始化。

MAE 验证重建 MSE 从 0.052329 降到 0.032924，最优点位于 epoch 15。

## FD001 测试结果

| 模型 | 视觉初始化 | RMSE | MAE | NASA Score | 参数量 |
|---|---|---:|---:|---:|---:|
| Temporal-only | — | 13.0132 | 9.3954 | 336.51 | 106,241 |
| Temporal + ViT | 随机 | 13.0041 | 9.1577 | 348.43 | 745,986 |
| Temporal + local MAE | 15轮同域掩码预训练 | 12.9099 | 8.9979 | 330.25 | 745,986 |
| Temporal + CNN | 随机、联合频谱融合 | **12.8423** | **8.9822** | **303.73** | 464,926 |

相对于随机 ViT，MAE 将 RMSE 降低约 0.72%，MAE 降低约 1.75%，NASA Score
降低约 5.22%。因此，在相同 ViT 架构和微调配置下，掩码重建预训练带来了小幅但一致
的测试改进。

轻量 CNN 在本轮仍然最好，不过 CNN 实验采用频谱变换与编码器联合训练，而 ViT/MAE
对比采用固定缓存，因此 CNN 与二者不是完全严格的单变量对照。严格结论应分成两层：

1. **ViT vs MAE 是受控对比**：MAE 初始化优于随机 ViT；
2. **CNN 是当前工程最优基线**：参数更少，且 FD001 三项测试指标均最好。

## 与论文 MAE 的边界

论文没有公开 MAE 的架构、预训练数据或 checkpoint。本复现使用的是小型、本地、同域
MAE，不能等同于论文所谓的预训练 MAE，也不能据此否定论文中 MAE 优于 CNN 的结论。
如获得作者 checkpoint 或明确模型标识，应直接替换当前视觉编码器并保持缓存和评估
流程不变。

## 产物

- 固定频谱缓存：`artifacts/spectrum_cache/FD001/split_seed42/`；
- MAE 预训练：`artifacts/mae_pretrain/FD001/seed42/`；
- 随机 ViT 微调：`artifacts/temporal_vit/FD001/seed42/`；
- MAE 微调：`artifacts/temporal_mae/FD001/seed42/`；
- CNN 实验：`artifacts/temporal_visual/FD001/seed42/`。
