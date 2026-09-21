# TMAF 方法保真审计（对照论文 III-D、式 (13)–(19)、表 II）

范围：只做「方法保真」逐项核对，不改任何模型结构/训练/数据处理，不做替代融合实验。
材料：本地论文 PDF（TS-MLLM，14 页，IEEE TBDATA 接收版）第 III-D 节、式 (13)–(19)、
表 II，以及摘要/引言/架构综述中关于 TMAF 的描述；实现为 `src/ts_mllm/tmaf.py`。

## 一、四件事逐项核对

### 1. Qwen 全局向量如何取出（F_LLM 的来源与聚合）

论文只给出：

> F_LLM = LLM(Concat[e_vis, F_text])  (13)

并描述为「outputting a guided representation F_LLM」「the integrated semantic
representation F_LLM encapsulates the global health context」「F_LLM represents the
entire window globally」。

**问题**：LLM 的输出是**逐 token 的隐状态序列**，而式 (15)/(16) 把 F_LLM 当作**单个
向量**（K_LLM = F_LLM W_K，V_LLM = F_LLM W_V）。论文**没有说明**如何把 LLM 的序列输出
归约为这个单向量：是 masked mean？末 token？还是某个特殊位置？

- 当前实现：`global_pooling="mean"`，即对所有有效文本 token（另含 1 个视觉前缀）做
  masked mean。另有 `last_text` 选项（排除视觉前缀取末文本 token）。
- 判定：**论文未明确（复现假设）**。不是「论文明确、当前遗漏」的操作。

### 2. 是否真的广播到每个 patch

论文在 III-D 中**明确、反复**写明：

> Since F_LLM represents the entire window globally, **we broadcast it along the
> temporal dimension**.
>
> ...the global context is **replicated along the temporal axis**, resulting in
> effective dimensions of K_LLM, V_LLM ∈ R^{N×Dk}.

- 当前实现：`semantic = global_context.expand(-1, N, -1)`，把单个全局向量复制 N 份。
- 判定：**忠实复现**。广播/复制是论文明文方法，当前实现与之完全一致。

### 3. 注意力 Q/K/V 的来源与维度

论文（式 (14)–(17)，表 II）：

| 量 | 来源 | 维度 |
|---|---|---|
| Q_TS = F_TS W_Q | 时间 patch 特征 | R^{N×Dk}，Dk=64（表 II "Attention Key"） |
| K_LLM = F_LLM W_K | 单个全局 F_LLM（再复制 N 份） | R^{N×Dk}=64 |
| V_LLM = F_LLM W_V | 单个全局 F_LLM（再复制 N 份） | 论文文字 R^{N×Dk}=64，但表 II "Attention Output"=**32** |
| A = Softmax(Q_TS K_LLM^T / √Dk) | — | — |

- Q/K 的来源与维度与实现一致（`query_projection 64→64`、`key_projection 1024→64`，
  √Dk=√64）。
- **Value 维度内部矛盾**：式 (16) 的文字把 V_LLM 写成 R^{N×Dk}（=64），但表 II 明确
  "Attention Output 32"。当前实现取 **32**（遵循表 II），并已在 `PAPER_ALIGNMENT_AUDIT.md`
  披露「正文称 K/V 都为 N×Dk 与 Value32 有矛盾，采用表 II 数值」。
- 判定：Q/K 忠实；Value=32 遵循表 II、与式 (16) 文字冲突（论文内部矛盾，非实现遗漏）。

### 4. 融合后的残差/归一化/回归路径

论文有两处相关表述：

> (III-C 末段→III-D) ...we employ a **residual connection** followed by a final
> linear projection W_out:
>
> F_fused = Concat[F_TS, F_attn] W_out = F_TS W_TS + F_attn W_attn  (18)/(19)
>
> By partitioning W_out into modality-specific sub-matrices (W_out = [W_TS; W_attn]),
> this formulation explicitly defines our **linear fusion-gate mechanism**. Unconstrained
> by strictly positive non-linear activations (e.g., Sigmoid), these gate weights
> naturally take both positive and negative values.

- **「残差」措辞与公式矛盾**：文字说 "residual connection"，但式 (18)/(19) 是
  `Concat + 线性投影`，且对时间项 F_TS 也乘了**可学习权重 W_TS**（并非恒等残差
  F_TS + F_attn）。论文自己随后把它定义为「linear fusion-gate」，并强调**无 sigmoid**、
  权重可取正负。
- 当前实现：`fusion_projection = Linear(temporal_dim + attention_output_dim, temporal_dim)`，
  即 Concat[F_TS, F_attn] W_out，无 sigmoid、无额外 `+F_TS`。**忠实于式 (18)/(19)**。
- **归一化**：III-D 融合路径未提任何 LayerNorm/normalization；实现亦无 —— 无遗漏。
- **回归头**：论文仅说「processed by the regression head」（表 II：Fusion MLP 512、
  Dropout 0.5）。未说明如何把 N 长度序列 F_fused 归约为标量，也未说明激活函数。
  当前实现：`mean pool → Linear(64→512)→GELU→Dropout(0.5)→Linear(512→1)`。
  mean pooling 与 GELU 是**复现假设**；512/dropout0.5 与表 II 一致。
- 判定：融合门忠实于公式；「残差」为论文内部措辞不一致（公式胜出）；回归头的
  pooling/激活为假设，表 II 的 512/0.5 一致。

## 二、核心发现：论文自身公式导致注意力退化

式 (15)/(16) 与 III-D 明文把 **K_LLM、V_LLM 定义为单个全局向量 F_LLM 复制 N 份**。
于是：

- K_LLM 的 N 行完全相同 ⇒ Q_TS K_LLM^T 的 N 列完全相同；
- Softmax 沿 Key 轴 ⇒ **每个 Query 的注意力权重恒为 1/N（均匀分布）**；
- F_attn = A V_LLM ⇒ **每个 patch 得到同一个全局 value 向量**。

即：论文自己的公式（广播单向量）在数学上**不可能**实现其行文所声称的
「selective search」「weighted by temporal relevance」「emphasize specific time steps
with corresponding anomalous fluctuations」。这是**论文内部矛盾**，也是当前复现
（忠实实现该公式）得不到 Fig. 6 模态增益的**最强候选根因**——但不是复现 bug。

当前 `global_broadcast` 的「Key 全同、注意力恒为 1/38」是**论文公式的字面结果**，
与 `TMAF_FIGURE12_AUDIT.md` 的 GPU 核验一致（注意力偏离 1/38 的最大值 0）。

## 三、内部矛盾与未明确项清单

| 编号 | 事项 | 性质 |
|---|---|---|
| C-1 | 注意力是「选择性检索（逐 token）」还是「广播单全局向量」 | **内部矛盾**：行文 vs 式 (14)–(17) |
| C-2 | 式 (16) V_LLM ∈ R^{N×Dk} vs 表 II "Attention Output 32" | **内部矛盾**：正文 vs 表 II |
| C-3 | 文字 "residual connection" vs 式 (18)/(19) 线性门（无恒等残差） | **内部矛盾**：措辞 vs 公式 |
| U-1 | F_LLM 如何从 LLM 序列输出归约为单向量（mean? 末 token?） | **未明确** |
| U-2 | 回归头如何把 N 长度 F_fused 归约为标量（pooling）与激活函数 | **未明确**（表 II 只给 512/0.5） |
| U-3 | 式 (13) F_LLM=LLM(...) 隐含的单向量化未定义 | **未明确** |

## 四、结论（路径 B）

论文**没有**明确一个「当前实现遗漏、补上即可」的操作。相反：

1. 式 (14)–(19) 与 III-D 明文（广播单全局向量）**已被忠实实现**；
2. 论文行文里的「选择性检索」**与它自己的公式直接冲突**；把它实现成逐 token 交叉
   注意力属于**架构修改（新假设）**，不是「论文明文遗漏的步骤」。

**因果归因的严谨性**：均匀注意力是「最强**候选**根因」，但**并非由消融直接证明的唯一
因果根因**。即使 softmax 恒为 1/N，式 (18)/(19) 的门控仍可把同一 LLM 全局 value 以
学习到的权重注入各 patch（一种与时间无关、非选择性的信息混合）；因此确切缺失的是论文
声称的「选择性检索」，而不是「多模态信息流整体为零」。消融只能支持「完整模型相对单模态
无稳定增益」，不能单独把该结论归结为均匀注意力这一个原因。

因此采用**路径 B**：

- **正式结论：按论文已公开设定，无法复现 Fig. 6 的多模态增益。**
  四子集、统一 batch32、三 seed 的输入级消融表（`FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.md`）
  已足够支撑「整体未复现」，且不依赖任何调参或替代融合。
- 停止对 TMAF/prompt/数据处理的进一步调优。
- 「逐 token K/V 交叉注意力」「非广播的全局/局部混合」「其他 F_LLM 归约方式」等，一律
  单列为**后续改进实验（架构修正假设）**，不得称为论文复现结果。

## 附：关键原文与式号索引

- 式 (13) F_LLM = LLM(Concat[e_vis, F_text])（p.6 左栏）
- 式 (14) Q_TS = F_TS W_Q；式 (15) K_LLM = F_LLM W_K；式 (16) V_LLM = F_LLM W_V
- 式 (17) A = Softmax(Q_TS K_LLM^T / √Dk)；F_attn = A V_LLM
- 式 (18) F_fused = Concat[F_TS, F_attn] W_out；式 (19) = F_TS W_TS + F_attn W_attn
- III-D 明文：「broadcast it along the temporal dimension」「replicated along the
  temporal axis, resulting in effective dimensions of K_LLM, V_LLM ∈ R^{N×Dk}」
- 表 II（TMAF）：Attention Key 64 / Attention Output 32 / Fusion MLP Units 512 /
  Dropout Rate 0.5；Batch 128 / lr 0.002 / 30 epochs（TMAF 统一参数）
- 摘要/架构综述：「treats each temporal feature as a query」「selective search for
  supportive visual cues」「weighted by temporal relevance」（与广播公式矛盾）
