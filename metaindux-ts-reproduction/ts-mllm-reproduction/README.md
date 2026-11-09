# TS-MLLM 非官方复现

复现论文 *TS-MLLM: A Multi-Modal Large Language Model-based Framework for Industrial
Time-Series Big Data Analysis*（DOI `10.1109/TBDATA.2026.3695338`）的 C-MAPSS
FD001–FD004 实验。**严格复现已停止，结论落定。**

## 当前状态（一句话）

按论文公开设定忠实实现后，**既达不到论文报告的精度，也复现不出 Fig. 6 的多模态增益**；
根因在论文 TMAF 融合公式自身的内部矛盾，而非实现错误。

## 核心结论

1. **主方案未达论文精度**（`main_protocol_v1`，batch128，测试 RMSE）：

   | 子集 | 归一化 | 复现 | 论文 |
   |---|---|---:|---:|
   | FD001 | 全局 Min-Max | 16.32 | 12.45 |
   | FD002 | 按工况 Min-Max | 17.09 | 14.22 |
   | FD003 | 全局 Min-Max | 23.43 | 11.97 |
   | FD004 | 按工况 Min-Max | 19.00 | 15.94 |

2. **Fig. 6 模态增益未复现**：在统一 batch32 + figure12_v1、三随机种子下，四个子集的
   完整多模态模型相对单模态消融均无稳定优势（详见
   `artifacts/FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.md`）。

3. **根因在论文公式**：TMAF（式 13–19）明文把 `K_LLM/V_LLM` 定义为单个全局向量沿时间
   复制 N 份，使 softmax 权重恒为 1/N，数学上无法实现论文行文所称的「选择性检索」。
   这是论文内部矛盾（`TMAF_METHOD_FIDELITY_AUDIT.md`），是「最强**候选**根因」，但现有
   实验不能证明它是唯一原因。

详见总结论：**[REPRODUCTION_CONCLUSION.md](REPRODUCTION_CONCLUSION.md)**。

## 文档导航

| 文档 | 内容 |
|---|---|
| [REPRODUCTION_CONCLUSION.md](REPRODUCTION_CONCLUSION.md) | **总结论**：已实现/假设、四子集结果、Fig.6 未复现、内部矛盾、停止理由 |
| [TMAF_METHOD_FIDELITY_AUDIT.md](TMAF_METHOD_FIDELITY_AUDIT.md) | 方法保真审计：论文式 (13)–(19) vs 实现、四件事逐项、路径 B |
| [FIXED_MAIN_PROTOCOL_RESULTS.md](FIXED_MAIN_PROTOCOL_RESULTS.md) | 固定主方案 `main_protocol_v1`（batch128）结果与来源 |
| [PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md) | 论文—实现逐项核对（含历史链接，部分指向 `archive/`） |
| [FOUR_SUBSET_MULTIMODAL_RESULTS.md](FOUR_SUBSET_MULTIMODAL_RESULTS.md) | 四子集旧/新归一化对照（历史结果，已被主方案取代） |
| [FD003_TOKEN_VS_BROADCAST.md](FD003_TOKEN_VS_BROADCAST.md) | 两种融合公式（广播 vs 逐 token）对比 |
| [FD003_BATCH_DIAGNOSTIC.md](FD003_BATCH_DIAGNOSTIC.md) | FD003 batch size 单变量诊断 |
| [TMAF_FIGURE12_AUDIT.md](TMAF_FIGURE12_AUDIT.md) | TMAF 方程核验与 figure12_v1 prompt 修复 |

消融与审计数据：

| 文件 | 内容 |
|---|---|
| `artifacts/FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.md` / `.json` | 四子集 Fig.6 输入级消融（batch32、三 seed） |
| `artifacts/FD003_INPUT_MODALITY_ABLATION_3SEED.md` / `.json` | FD003 三 seed 输入级消融 |
| `configs/main_protocol_v1.json` | 固定主方案配置（锁定来源 hash） |

## 关键实验参数

- 数据：FD001–004，14 有效传感器（删 1/5/6/10/16/18/19），Min-Max（式 20），窗口 40，
  RUL 上限 125，样本/验证滑窗 stride 50，split_seed 42。
- 时间分支：Patch 4 / stride 1 / dim 64 / 单头 / 2 层，`F_TS=Encoder(Z0)`。
- 视觉：RP/STFT/CWT → 114×114 → MAE 128 维；文本：96 维 DKE，最大 512 token。
- TMAF：`global_broadcast`（论文式 14–19 字面实现），Key 64 / Output 32 / MLP 512 /
  dropout 0.5；batch 128、lr 0.002、30 epochs。
- 归一化：FD001/FD003 全局 Min-Max，FD002/FD004 按工况 Min-Max（补充预处理假设）。

**未公开部分的复现假设**（`F_LLM` 归约方式、回归头结构、对齐训练目标、主模型型号、
MAE 权重、prompt 内容等）完整披露于 `REPRODUCTION_CONCLUSION.md` §二。

## 历史归档

早期探索与过程记录（prompt 迭代、归一化实验、基线对比、逐项审计）已移入
[`archive/`](archive/)，仅作追溯，不再作为当前结论依据。若后续继续探索（逐 token
注意力、非广播融合、其他池化等），应在 `artifacts/improvements/` 下另起，与严格复现
产物完全隔离，并标注「改进实验，非论文复现」。

## 复现环境

- 运行示例见 `scripts/`（幂等脚本，离线加载 Qwen 需 `HF_HUB_OFFLINE=1`）。
- 测试：`pytest tests/`。
