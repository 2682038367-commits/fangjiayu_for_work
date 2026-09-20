# TS-MLLM 非官方复现

本目录用于复现论文 *TS-MLLM: A Multi-Modal Large Language Model-based
Framework for Industrial Time-Series Big Data Analysis*。

主方案已固定，见[FIXED_MAIN_PROTOCOL_RESULTS.md](FIXED_MAIN_PROTOCOL_RESULTS.md)及
[configs/main_protocol_v1.json](configs/main_protocol_v1.json)：FD001采用修复后的stride50 MAE链路，
FD003保留已核验链路；FD002/FD004采用按工况Min-Max＋原统计文本A。
四集Test RMSE16.3233/17.0862/23.4251/18.9958，其余模型和训练参数不变。
FD001独立GPU回放已补做通过；旧结果缺少初始化hash，不宣称已核验同初始化。
下文为历史状态，主报告以固定清单为准。

FD001/FD003来源已核对：FD003无需重跑；FD001旧MAE仍使用stride1/uint8谱图，
只重建其MAE及下游对齐/缓存/TMAF，保留数据和时间权重。GPU入口
`scripts/repair_fd001_mae_protocol.py`，见[FD001_FD003_PROVENANCE_AUDIT.md](FD001_FD003_PROVENANCE_AUDIT.md)。
代码验证完成，新训练尚未运行。

工况归一化A的低成本文本输出消融已准备，入口
`scripts/run_condition_text_ablation.py`；详见[CONDITION_TEXT_ABLATION.md](CONDITION_TEXT_ABLATION.md)。
复用A缓存，只保留视觉前缀输出重训TMAF，尚未启动训练，不覆盖A/B结果。

当前主推进方案保留按工况Min-Max（文献支持的补充预处理，非原文明文方法）。
新文本方案对照入口为 `scripts/run_condition_knowledge_comparison.py`，
见[CONDITION_KNOWLEDGE_COMPARISON.md](CONDITION_KNOWLEDGE_COMPARISON.md)。
原统计文本结果保留，新有序工况/通用知识prompt明确注明校正后的传感器尺度，
全窗口最长458 tokens；代码验证完成，尚未训练B。以下为历史阶段记录。

下一步全局向量提取对照已准备：复用有序工况缓存，只比较平均向量与最后有效文本token，
见[GLOBAL_POOLING_COMPARISON.md](GLOBAL_POOLING_COMPARISON.md)。GPU入口
`scripts/run_global_pooling_comparison.py`；当前尚未运行训练。

当前 prompt 已升级为有序工况序列 `operating_sequence_v2`，见
[ORDERED_OPERATING_PROMPT.md](ORDERED_OPERATING_PROMPT.md)。全部窗口最大427 tokens，
48项测试通过；模型及训练设置不变，旧v1产物保留，新入口输出到独立v2目录。

当前主流程：按用户决定恢复全局传感器 Min-Max，不做工况归一化；
新增 NASA 来源的通用知识及当前窗口工况摘要 prompt，见
[GLOBAL_KNOWLEDGE_PROMPT.md](GLOBAL_KNOWLEDGE_PROMPT.md)。
执行 `PYTHONPATH=src ../.venv/bin/python scripts/run_global_knowledge_tmaf.py`，
默认在 GPU 重建 FD002/FD004 对齐和缓存，再训练完整 TMAF；本次尚未运行训练。
新增 prompt 属于论文未公开部分的复现假设。下文“最新”段落均保留为历史阶段记录。

最新：FD002/FD004按工况归一化的MAE、对齐、float32频谱及完整Qwen缓存已在GPU重建，
来源、行映射、完整文本和文件checksum通过，见
[CONDITION_MULTIMODAL_REBUILD.md](CONDITION_MULTIMODAL_REBUILD.md)。本阶段未训练TMAF，旧缓存保留。

最新FD004同协议归一化验证见[FD004_CONDITION_NORMALIZATION_RESULTS.md](FD004_CONDITION_NORMALIZATION_RESULTS.md)：
全局/条件化时间Test RMSE44.7718/20.7756；两组初值相同、248条预测及独立GPU核验通过。
FD002归一化验证见[CONDITION_NORMALIZATION_RESULTS.md](CONDITION_NORMALIZATION_RESULTS.md)：
保持相同初值和论文明确训练参数，按工况Min-Max时间基线Val RMSE21.4168、
Test RMSE18.7242（全局43.9029），预测已摆脱近似常数。属于未明确拟合范围的
复现假设；两集均已验证时间基线并重建新多模态缓存，完整TMAF尚未按此重跑。

最新：FD002–FD004完整GPU多模态流程已完成，保留默认输出bias和B目标尺度，
不运行稳定性实验。四集主流程结果见[FOUR_SUBSET_MULTIMODAL_RESULTS.md](FOUR_SUBSET_MULTIMODAL_RESULTS.md)。
完整TMAF测试RMSE为15.2218/44.3397/23.4251/44.9046，仍未复现论文性能；
FD002/FD004时间分支近似常数。FD001与其余子集MAE预训练来源尚不完全一致，报告明确披露。

论文 DOI：`10.1109/TBDATA.2026.3695338`。

**状态声明：现有结果均包含复现假设，尚无完全忠实的最终多模态主结果。**
依据与未决项见 [PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md)。分阶段小学习率
训练是额外实验，不是论文明确方法。下文TMAF默认命令复用legacy缓存与stride1协议，
不能将修正后的参数套用到旧产物解释旧指标。

优先级1已完成：四集window40/sample-stride50独立数据产物，见
[DATA_STRIDE50_REBUILD.md](DATA_STRIDE50_REBUILD.md)。后续应使用这些新数据，不复用旧
stride1频谱/Qwen缓存。Patch stride始终保持1。

优先级2/3已完成FD001：96维DKE实际接入Qwen、线性视觉projector及桥接对齐训练，见
[SVLMA_ALIGNMENT_RESULTS.md](SVLMA_ALIGNMENT_RESULTS.md)。默认新缓存入口读取这些新权重。
线性桥接与监督目标是明确编号的复现假设，不是论文原始已公开方案。

优先级4/5已完成FD001：完整512文本缓存、float32新频谱及新stride50时间分支/字面
TMAF训练，见[AUDITED_FD001_RESULTS.md](AUDITED_FD001_RESULTS.md)。新TMAF RMSE=24.0989，
尚未达到论文效果；不要使用旧12.5539替代此协议结果。

目标尺度A/B单变量验证见[TARGET_SCALE_RESULTS.md](TARGET_SCALE_RESULTS.md)。同一初始化
和明确训练参数下，B=RUL/125在验证集优于原始尺度，测试RMSE21.5395；这是独立尺度
假设实验。现按用户决定，后续所有RUL回归训练统一采用B：训练目标RUL/125、
MSE；验证、测试、CSV及曲线均乘125恢复周期单位。新检查点记录target_divisor=125。
这属于论文未明确的复现假设；历史结果及旧检查点不变。
FD001已按B重跑完整字面TMAF，验证RMSE16.8625、测试RMSE15.2218；
详见[TMAF_SCALE_B_RESULTS.md](TMAF_SCALE_B_RESULTS.md)。单seed改善不等于稳定提升，
且两阶段训练预算不同，不能仅据此归因Qwen。下一步检查论文明确的回归头结构，
最后只对未说明的初始化方案做单变量对照。

回归头与初始化核验已完成，见[HEAD_INITIALIZATION_AUDIT.md](HEAD_INITIALIZATION_AUDIT.md)。
512/dropout0.5明确项一致，未更改架构或训练权重；完整结构仍标为假设。
新融合头初始输出偏低作为输出bias初始化的单变量候选，后续实验结果见下。

输出bias单变量实验已完成，见[BIAS_INITIALIZATION_RESULTS.md](BIAS_INITIALIZATION_RESULTS.md)。
均值bias验证RMSE16.5992（默认16.8625），测试RMSE18.0168（默认15.2218）；
按验证标准实验选择均值bias，但不宣称泛化改善，未自动改CLI默认。建议补配对多seed。

目前完成 C-MAPSS 数据管线和纯时序 Patch Transformer。数据处理的实现假设、
验收方式和已知论文歧义见 [DATA_PIPELINE.md](DATA_PIPELINE.md)，FD001 稳定性实验见
[TEMPORAL_BASELINE.md](TEMPORAL_BASELINE.md)，四个子集的正式结果见
[TEMPORAL_RESULTS.md](TEMPORAL_RESULTS.md)，频谱与 CNN 视觉基线见
[VISUAL_BASELINE.md](VISUAL_BASELINE.md)，CNN/ViT/MAE 对比见
[VISION_ENCODER_COMPARISON.md](VISION_ENCODER_COMPARISON.md)，冻结 Qwen 多模态分支和
token 缓存见 [QWEN_MULTIMODAL_CACHE.md](QWEN_MULTIMODAL_CACHE.md)。
TMAF 的论文公式对应、实现假设与 FD001 结果见 [TMAF_RESULTS.md](TMAF_RESULTS.md)。
FD001 的 token 消融结果及其模态解释限制见
[TMAF_ABLATION_RESULTS.md](TMAF_ABLATION_RESULTS.md)。
三随机种子稳定性结果见 [TMAF_STABILITY_RESULTS.md](TMAF_STABILITY_RESULTS.md)。
分阶段双学习率训练见 [TMAF_STAGED_RESULTS.md](TMAF_STAGED_RESULTS.md)。

## 目录结构

```text
ts-mllm-reproduction/
├── data/                 # 数据放置说明；原始数据不重复提交
├── scripts/              # 数据审计及后续实验入口
├── src/ts_mllm/          # 可复用实现
├── tests/                # 单元测试
├── DATA_PIPELINE.md      # 第二阶段说明
└── pyproject.toml
```

## 安装

```bash
cd ts-mllm-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

当前工作区已经安装了所需依赖，也可以直接使用上级目录的环境：

```bash
cd ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/audit_ts_mllm_data.py
PYTHONPATH=src ../.venv/bin/pytest
```

## 训练纯时序基线

论文明确参数为窗口 40、Patch 4、Patch 步长 1、模型维度 64、单头注意力、
batch size 128、学习率 0.002 和 30 epochs。未公开部分固定为两层
Transformer、平均池化、Adam 和 MSE：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_temporal.py --dataset FD001
```

输出保存在 `artifacts/temporal/FD001/seed42/`，包括验证集最优 checkpoint、
训练历史、测试指标、逐发动机预测 CSV 和无需 Matplotlib 的 SVG 预测曲线。
指标同时保存原始预测值和裁剪到 `[0, 125]` 后的结果，主结果使用裁剪值。
稳定性实验只改变 `--seed`，并保持默认的 `--split-seed 42`，从而确保不同运行
使用同一组训练/验证发动机。

FD001 验收通过后，可并行训练剩余三个子集：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/run_temporal_remaining.py
```

训练 FD001 的 RP/STFT/CWT + 轻量 CNN 视觉基线：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_visual.py --dataset FD001
```

默认加载纯时序最优 checkpoint 后联合训练，并在
`artifacts/temporal_visual/FD001/seed42/` 保存频谱示例、最优模型和测试结果。

## 生成 Qwen 多模态 token 缓存

需要可见的 NVIDIA GPU 和已经下载到本地的 `Qwen/Qwen3-0.6B` 权重：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/cache_qwen_multimodal.py
```

该命令冻结 MAE 视觉编码器与 Qwen，只训练 MAE-to-Qwen projector，然后生成
FD001 train/val/test 的定长 token 缓存及完整性 manifest。CUDA 不可用时会直接
终止，不会回退到 CPU。

本次审计后，新Qwen入口默认改为线性projector、文本512、完整文本缓存、train/val
stride50及mae_pretrain encoder，输出qwen_audited目录。旧stride1谱图不能直接使用；
96维DKE现通过线性桥接实际接入，新默认从核验后的导出窗口在线计算谱图并加载新
对齐权重，不需要旧谱图缓存。桥接与projector目标仍未由论文确定，采用披露假设，
此命令不是已完整忠实的方法配方。旧缓存详情见历史报告。

## 训练 TMAF

直接读取已经生成的 Qwen token 缓存，并在 GPU 上训练时间中心交叉注意力融合层：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_tmaf.py
```

默认从纯时序最优 checkpoint 初始化时间分支，使用论文的 attention key 64、
attention output 32、融合 MLP 512、dropout 0.5、batch size 128、学习率 0.002
和 30 epochs。结果保存在 `artifacts/tmaf/FD001/seed42/`。

执行仅视觉、仅文本和跨样本打乱 token 的独立训练消融：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/run_tmaf_ablations.py
```

注意：仅文本输出已在 Qwen 前向时融合视觉前缀，因此这是输出 token 消融。
严格无视觉的输入模态消融需要重新生成缓存。

## 当前状态

- [x] FD001–FD004 读取与完整性校验
- [x] 14 个有效传感器选择
- [x] 训练集专属 Min-Max 归一化
- [x] RUL 上限 125 与固定长度 40 滑窗
- [x] 按发动机划分和少样本抽样
- [x] 验证发动机全寿命滑窗评估
- [x] 官方测试终点映射与短序列填充
- [x] 纯时序 Patch Transformer
- [x] RP/STFT/CWT 三通道频谱图
- [x] 轻量 CNN 视觉基线（128 维）
- [x] ViT 与 MAE 模型及训练入口
- [x] MAE 自监督预训练和 CNN/ViT/MAE 指标对比
- [x] Qwen3-0.6B 动态文本、视觉前缀与冻结推理
- [x] FD001 多模态 token 缓存及完整性审计
- [x] TMAF 融合层与 FD001 完整训练
- [x] FD001 TMAF 输出 token 消融与打乱负对照
- [x] FD001 TMAF 三随机种子稳定性实验
- [x] 时间分支冻结热身与双学习率联合训练（FD001 seed42）
- [ ] 严格输入模态消融与少样本实验

这是非官方复现。论文和 arXiv 页面目前没有公开源代码，因此所有补充假设都应在
实验报告中明确记录。
