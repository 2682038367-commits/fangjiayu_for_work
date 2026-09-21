# FD002、FD004条件化归一化特征链重建

已在RTX4060Ti上独立重建两个子集的MAE、对齐、RP/STFT/CWT频谱和完整Qwen缓存。
输入来自验证有效的训练集拟合、六工况Min-Max数据。本次未训练时间回归或TMAF，
没有新的RMSE/MAE/Score；旧全局归一化模型和缓存均未覆盖。

## 配置与假设

- window40/sample-stride50、14传感器，split_seed42/seed42。
- MAE预训练15轮、batch128、AdamW/lr0.001、mask0.75，本地compact MAE128维；
  在线GPU float32频谱、均匀sensor fusion用于预训练。这是未公开预训练日程的假设。
- 对齐30轮、batch128、Adam/lr0.002；冻结新MAE和Qwen3-0.6B，训练DKE96/位置编码、
  96→1024线性桥接、128→1024线性视觉projector及谱图sensor fusion。
- 沿用训练词表SVD初始化、teacher token embedding MSE与视觉prefix→masked teacher
  mean MSE等权对齐，不使用RUL标签训练对齐目标；这些目标与桥接仍为复现假设。
- Qwen BF16冻结推理，缓存float16。max_text_tokens512，保留1个视觉prefix与全部
  有效文本输出，容量513；无效状态置零，保存bool mask，不强制实际文本为512。
- 谱图为float32、scale1、形状[N,3,114,114]，不使用旧uint8频谱或旧sensor权重。
- 按工况Min-Max本身也是论文未明确拟合范围的复现假设；不称已证实作者实现。

## 数量与验收

| 子集 | train/val/test | Qwen容量（单样本） | 频谱（单样本） |
|---|---|---|---|
| FD002 | 812/195/259 | [513,1024] | [3,114,114] |
| FD004 | 938/220/248 | [513,1024] | [3,114,114] |

每集MAE、对齐、缓存的data_manifest_sha256均匹配新的条件化数据；检查MAE权重
SHA、alignment权重SHA和全部Qwen/频谱缓存文件SHA。谱图shape/dtype/finite检查通过。
缓存token/mask/target/unit/cycle、padding归零、行映射、每行完整prompt文本长度
检查通过；对齐阶段亦进行了真实Qwen验证/测试前向，结果有限。
44项模块测试、compileall通过。这些是工程完整性检查，不证明语义有效或RUL提升。

## 独立目录

以下每个目录分别包含FD002及FD004：

- MAE：`artifacts/mae_condition_minmax/{dataset}/stride50_seed42/`
- 对齐：`artifacts/alignment_condition_minmax/{dataset}/stride50_seed42/`
- 频谱：`artifacts/spectrum_condition_minmax/{dataset}/stride50_seed42/`
- Qwen：`artifacts/qwen_condition_minmax/{dataset}/seed42/`
- 阶段日志和核验汇总：`artifacts/condition_multimodal_rebuild_seed42/`

MAE/对齐含验证最优best.pt及history/result；频谱/Qwen含各split数组及来源manifest。
summary.json记录所有路径和缓存、谱图检查结果。重建脚本复用已完成阶段并核验来源，
未完成训练不会回退CPU，也不会静默覆盖旧模型。

```bash
PYTHONPATH=src ../.venv/bin/python scripts/rebuild_condition_multimodal.py
```

## 下一步

批量训练与GPU重放入口（默认依次FD002、FD004；不重建特征）：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_condition_tmaf.py
```

可加`--datasets FD002`只跑一集，或`--dry-run`仅核对来源并打印命令。
结果保存在`artifacts/tmaf_condition_minmax/{dataset}/global_broadcast_stride50_B_seed42/`。
已完成结果保留并重放核验；已有部分checkpoint但未完成的运行拒绝覆盖。

完整TMAF必须配套使用：

1. `artifacts/data_condition_minmax/split_seed42/{dataset}`窗口；
2. `artifacts/temporal_condition_minmax/{dataset}/stride50_B_seed42/best.pt`时间权重；
3. `artifacts/qwen_condition_minmax/{dataset}/seed42`新Qwen缓存。

保持目标RUL/125、默认输出bias、global_broadcast、30轮/batch128/lr0.002、冻结0轮。
数据SHA校验会拒绝混入旧全局归一化窗口、旧时间权重或旧Qwen缓存。
