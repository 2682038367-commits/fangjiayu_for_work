# FD003 输入级模态消融 · 三随机种子最小重复验证

对应论文 Fig. 6 的 `w/o Visual`、`w/o Textual` 概念。在已完成的 seed 42 输入级
消融（`FD003_INPUT_MODALITY_ABLATION.md`）基础上，固定 FD003 数据划分
（split_seed=42）与时间分支的同一初始 checkpoint，仅更换两个随机种子（52、62），重跑
完整 / 去视觉 / 去文本三组。未改动 TMAF 结构、prompt 或数据处理。

## 控制变量

三组共用：数据 manifest `b0cddcfd…`、同一初始时间分支 checkpoint `a1002727…`
（`fd003_batch_diagnostic/seed42/batch32/best.pt`，训练中联合微调）、`figure12_v1`
prompt、`global_broadcast` + `mean`、batch=32、30 epochs、Adam lr=0.002、无冻结、
目标尺度 B=RUL/125、seed/split 分离。三组 Qwen 缓存为确定性推理产物，跨 seed
复用不变。时间分支与 TMAF/回归头共享同一初始 checkpoint，但
`temporal_learning_rate=0.002` 且无冻结，训练中时间分支随 seed 联合微调；因此
seed 波动来源包括 TMAF、回归头与时间分支的训练轨迹。三组之间仍公平（同一起点），
不宜把波动仅归因于融合层初始化。

- full：Qwen 输入 = 视觉前缀 + DKE 文本（165–166 有效 token）
- w/o Visual：Qwen 输入 = 仅 DKE 文本（164–165 有效 token，无频谱/视觉编码器）
- w/o Textual：Qwen 输入 = 仅投影视觉前缀（恰好 1 个有效 token，无 prompt 分词）

## 验证 RMSE（按验证集选 checkpoint）

| Qwen 输入 | seed 42 | seed 52 | seed 62 | 均值 | 样本标准差 | 极差 |
|---|---:|---:|---:|---:|---:|---:|
| full (temporal + visual + textual) | 17.490 | 18.689 | 18.099 | **18.092** | 0.600 | 1.199 |
| w/o Visual (temporal + textual)     | 17.505 | 18.537 | 17.015 | **17.685** | 0.777 | 1.522 |
| w/o Textual (temporal + visual)     | 17.288 | 17.826 | 17.755 | **17.623** | 0.293 | 0.539 |

验证 MAE 均值：full 11.539 / w/o Visual 12.431 / w/o Textual 12.371。
验证 Score 均值：full 1026.6 / w/o Visual 856.3 / w/o Textual 851.1。

测试 RMSE（仅作保留的泛化观察，不用于选模型）：full 21.513 ± 2.157 /
w/o Visual 21.117 ± 1.560 / w/o Textual 21.650 ± 1.543。

## 结论

**当前严格实现下，Fig. 6 的模态增益未复现。** 三个种子验证 RMSE 均值里，完整
模型（18.092）反而比去视觉（17.685）差 0.41、比去文本（17.623）差 0.47，且该
差值落在各模态自身的 seed 波动范围（std 0.29–0.78）之内。换言之：既没有观察到
视觉或文本提供稳定、独立的验证增益，单次运行的模态间差异（seed 42 时最大仅
0.217）也不足以与初始化/训练顺序波动（同一模态跨 seed 极差可达 0.54–1.52）区分。

这与此前单 seed 观察一致，且已通过三 seed 均值确认：seed 42 一次运行中完整模型
并非最优（去文本反而更低），三 seed 均值同样不支持完整模型占优。因此不把
Fig. 6 的模态贡献视为已在当前严格协议下复现，也不据此继续在 FD003 上调 TMAF、
prompt 或数据处理。

## 解释边界

- 三 seed 只能作重复测量，不能作统计显著性检验；且未按相同 52/62 seed 重训纯
  时序对照，无法把总方差精确归因到融合层、缓存或 projector 中的单一组件。
- 视觉 projector 的上游对齐训练以文本 teacher embedding 为目标（论文未公开的
  复现假设），故 `w/o Textual` 仅表示 Qwen 推理输入无文本，不表示视觉表示在训练
  阶段完全无文本监督。
- CUDA memory-efficient attention 存在非严格确定性提示，本报告是训练稳定性而非
  位级复现保证。
- 测试集数值仅保留，不能用于选择模型或判定模态贡献。

## 产物

- 逐 seed 逐模态结果：`artifacts/fd003_input_ablation/seed{52,62}/tmaf_{full,no_visual,no_text}/`
  （seed 42 复用既有 `tmaf_batch32` 与 `tmaf_no_visual/tmaf_no_text`）
- 汇总：`artifacts/FD003_INPUT_MODALITY_ABLATION_3SEED.json`
- 复跑入口：`scripts/run_fd003_input_ablation_repeat.py`（幂等，已有结果经协议校验后复用）

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_fd003_input_ablation_repeat.py
```
