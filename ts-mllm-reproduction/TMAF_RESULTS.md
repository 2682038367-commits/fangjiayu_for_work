# TMAF 实现与 FD001 结果

历史假设配置记录，不是完全忠实复现主结果；逐项差异见
[PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md)。尤其token K/V不是正文global broadcast
的字面实现，旧Qwen缓存也不满足线性projector/文本512/独立96维DKE。

本阶段直接读取冻结 Qwen 生成的 `[32, 1024]` token 缓存，实现论文第 III-D 节的
Temporal-centric Multi-modal Attention Fusion（TMAF），并在 RTX 4060 Ti 上完成
FD001 的 30 epochs 训练。

## 结构对应

论文公式 (14)–(19) 对应的实现如下：

- `FTS`：Patch Transformer 输出，形状 `[B, 38, 64]`；
- `QTS = FTS WQ`：Query 维度 64；
- `KLLM = FLLM WK`：32 个 Qwen token 分别投影到 64 维；
- `VLLM = FLLM WV`：32 个 Qwen token 分别投影到 32 维；
- `A = Softmax(QK^T / sqrt(64))`：每个时间 patch 主动检索多模态 token；
- `Fattn = A V`：得到 `[B, 38, 32]` 上下文；
- `Concat[FTS, Fattn] Wout`：无 Sigmoid 的线性融合门；
- 时间维平均池化后使用 `64 → 512 → 1` 回归 MLP，Dropout 为 0.5。

时间分支从纯时序 FD001 最优 checkpoint 初始化；缓存的 Qwen 特征保持冻结。论文
对 `FLLM` 是单个全局向量还是 token 序列存在描述歧义。若把同一个全局向量沿时间
复制，交叉注意力会退化为均匀权重，因此本复现保留缓存中的 32 个语义 token 作为
Key/Value，这也符合“每个时间特征主动检索视觉和文本细节”的设计目标。

## 训练设置

- Batch size：128；
- Learning rate：0.002；
- Epochs：30；
- Attention key：64；
- Attention output：32；
- Fusion MLP：512；
- Dropout：0.5；
- 最优模型选择：验证集 RMSE；
- GPU：NVIDIA GeForce RTX 4060 Ti。

运行命令：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_tmaf.py
```

## FD001 结果

| 模型 | RMSE | MAE | Score |
|---|---:|---:|---:|
| 纯时序 Patch Transformer | 13.0132 | 9.3954 | 336.51 |
| TMAF（本阶段） | 13.3737 | 9.6078 | 385.74 |
| 论文 TS-MLLM | 12.45 | 未在主表给出 | 233.40 |

TMAF 最佳验证轮次为第 7 轮，验证 RMSE 为 12.4755；测试预测标准差为 40.0013，
说明没有退化为常数；测试集恰好输出 100 个结果，训练总耗时约 92.42 秒。

当前结论是：TMAF 数据流、交叉注意力和训练链路已经完整打通，但单次 seed=42 的
结果尚未超过纯时序基线，也未达到论文的 12.45。可能影响因素包括当前缓存只保留
32 个 token（论文最大长度为 512）、Qwen 与论文未明确的特征聚合方式、projector
监督目标以及联合训练策略。下一步应先做融合消融与多随机种子稳定性实验，再决定
是否扩展 token 缓存或调整分阶段训练。

产物位于 `artifacts/tmaf/FD001/seed42/`，包含验证集最优 checkpoint、完整训练历史、
测试指标、逐发动机预测 CSV 和 SVG 曲线。
