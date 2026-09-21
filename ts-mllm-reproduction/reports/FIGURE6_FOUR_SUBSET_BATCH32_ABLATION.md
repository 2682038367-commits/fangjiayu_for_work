# FD001–FD004 输入级模态消融 · 统一 batch32 + figure12_v1 · 三 seed

对应论文 **Fig. 6**（`w/o Visual`、`w/o Textual`）的严格输入级消融，四子集统一协议、
按验证集汇总。目标：区分「FD003 特例」与「当前复现整体未能得到多模态增益」。

## 统一协议（保证四子集可直接比较）

- 所有子集：`batch_size=32`、30 epochs、Adam lr=0.002、无冻结（`temporal_learning_rate=0.002`）、
  样本/验证步长 50、`global_broadcast` + `mean`、目标尺度 B=RUL/125、seed/split=42。
- 同一 `figure12_v1` 文本协议（Task/Dataset Feature/Dataset Describe）。
- 同一输入级消融定义：`full`（视觉前缀 + DKE 文本）、`w/o Visual`（仅 DKE 文本，
  有效 token 164–165）、`w/o Textual`（仅投影视觉前缀，有效 token 恰为 1）。不是
  Qwen 输出后的 token 屏蔽。
- 完整/去视觉/去文本三组共享同一**初始时间权重**（每子集一个 batch32 时间 checkpoint）；
  训练中时间分支随 seed 联合微调，故 seed 波动含 TMAF、回归头与时间分支的训练轨迹。
- 每个子集固定其主实验数据与归一化：FD001/FD003 全局 Min-Max，FD002/FD004 按工况 Min-Max。
- 三个随机 seed（42/52/62），仅按验证集选 checkpoint 与汇总。

## 结果（验证 RMSE，均值 ± 样本标准差）

| 子集 | full (T+V+Txt) | w/o Visual (T+Txt) | w/o Textual (T+V) |
|---|---:|---:|---:|
| FD001 | **17.708 ± 0.278** | 17.773 ± 0.421 | 17.764 ± 0.403 |
| FD002 | 17.870 ± 0.403 | 18.028 ± 0.468 | **17.719 ± 0.192** |
| FD003 | 18.092 ± 0.600 | 17.685 ± 0.777 | **17.623 ± 0.293** |
| FD004 | 17.178 ± 0.633 | **16.917 ± 0.454** | 17.088 ± 0.350 |

完整模型相对最佳单模态消融的验证 RMSE 差值：

| 子集 | full − 最佳消融 |
|---|---:|
| FD001 | −0.056（full 微弱占优，远小于自身 seed 波动 0.278） |
| FD002 | +0.151（去文本更优） |
| FD003 | +0.470（去文本更优） |
| FD004 | +0.261（去视觉更优） |

测试 RMSE（仅作保留的泛化观察，**不用于选模型或判定模态贡献**）：

| 子集 | full | w/o Visual | w/o Textual |
|---|---:|---:|---:|
| FD001 | 17.228 ± 1.640 | 17.557 ± 2.109 | 17.975 ± 1.345 |
| FD002 | 17.091 ± 0.673 | 18.063 ± 0.634 | 17.379 ± 1.159 |
| FD003 | 21.513 ± 2.157 | 21.117 ± 1.560 | 21.650 ± 1.543 |
| FD004 | 18.380 ± 0.322 | 18.450 ± 0.282 | 18.251 ± 0.404 |

## 结论

**在当前严格实现下，Fig. 6 的模态增益未在四个子集的任何一个上复现。** 四子集验证
RMSE 均值里，完整多模态模型都没有相对单模态消融的稳定优势：FD002/FD003/FD004 上
去掉一种模态反而更低（0.15–0.47 周期）；FD001 上完整模型仅以 0.056 的幅度微弱占优，
远小于其自身 seed 波动（0.278），无法与初始化/训练轨迹波动区分。

因此这不是 FD003 的特例，而是**当前复现整体未能获得多模态增益**。这与此前单 seed 的
FD003 观察一致，并已通过四子集统一协议 + 三 seed 均值确认。

## 解释边界

- 三 seed 只是重复测量，不能作统计显著性检验；每模态每子集仅 3 个样本。
- `global_broadcast` 的公式字面实现使 38 个 Key 完全相同、softmax 恒为 1/38，时间
  Query 无法做逐 patch 语义选择；这是论文明文公式的结构性退化，是本复现中多模态
  未带来增益的候选根因，但本消融本身不能证明因果。
- 视觉 projector 的上游对齐训练以文本 teacher embedding 为目标（论文未公开的复现
  假设），故 `w/o Textual` 仅表示 Qwen 推理输入无文本，不表示视觉表示在训练阶段
  完全无文本监督。
- FD002/FD004 采用按工况 Min-Max（文献支持的补充预处理，非原文明文方法）。
- CUDA memory-efficient attention 存在非严格确定性提示；本表是训练稳定性，不是位级复现保证。

## 产物

- 汇总 JSON：`artifacts/FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.json`（36 次运行的逐 seed
  指标 + 四子集聚合）
- 新资产（FD001/FD002/FD004）：`artifacts/figure12_batch32_ablation/<ds>/{temporal,
  qwen_no_visual,qwen_no_text,tmaf_full,tmaf_no_visual,tmaf_no_text}/`
- FD003 复用既有 batch32 figure12_v1 三 seed 输入消融（`fd003_figure12_v1` 与
  `fd003_input_ablation`）。
- 复跑入口：`scripts/run_figure12_batch32_ablation.py`（幂等，已离线加载 Qwen）。

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_figure12_batch32_ablation.py   # 全量（幂等）
PYTHONPATH=src ../.venv/bin/python scripts/run_figure12_batch32_ablation.py --subsets FD002
```
