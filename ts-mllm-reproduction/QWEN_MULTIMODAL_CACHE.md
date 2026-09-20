# Qwen 多模态分支与 token 缓存

本文是历史简化配置记录（MLP projector、文本128、缓存32、stride1）。不是完全忠实
复现；修正后的入口默认值与本文不同，旧产物未重建。最新核对见
[PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md)。

本阶段在 FD001 上接入冻结的 `Qwen/Qwen3-0.6B`，并使用 RTX 4060 Ti 完成
projector 训练和全量多模态 token 缓存。Qwen 以 BF16 推理，MAE 视觉编码器与
Qwen 参数均冻结，只有视觉 projector 参与训练。

## 数据流

1. 将每个 `[40, 14]` 窗口转换成不包含 RUL 标签的动态文本描述；
2. 从 RP/STFT/CWT 三通道图像提取 128 维 MAE 表征；
3. 用两层 MLP projector 将视觉表征映射到 Qwen 的 1024 维 embedding 空间；
4. 将映射后的视觉向量作为一个前缀 token，拼到文本 embedding 前；
5. 通过冻结的 Qwen3-0.6B 前向传播；
6. 每个样本保存视觉前缀输出和最后 31 个有效文本输出，得到 `[32, 1024]`。

论文没有完整公开 projector 监督目标。本复现固定采用动态 prompt token embedding
的 masked mean 作为对齐目标，并以 MSE 训练 projector；这是明确记录的补充实现
假设，不使用真实 RUL，避免标签泄漏。

## 运行方式

此阶段必须使用 CUDA；代码在 CUDA 不可见时会直接报错，不会静默回退到 CPU。

```bash
PYTHONPATH=src ../.venv/bin/python scripts/cache_qwen_multimodal.py
```

默认从 `artifacts/temporal_mae/FD001/seed42/best.pt` 加载已经监督微调过的
MAE 视觉编码器，并从本地 Hugging Face 缓存加载 Qwen 权重。

## FD001 实际结果

- GPU：NVIDIA GeForce RTX 4060 Ti；
- Qwen：Qwen3-0.6B，BF16，冻结；
- projector：5 epochs，验证 MSE 从 `0.614700` 降至 `0.000695`；
- 总运行时间：约 `111.70 s`；
- train：`[13407, 32, 1024]`，float16；
- val：`[3324, 32, 1024]`，float16；
- test：`[100, 32, 1024]`，float16。

缓存位于 `artifacts/qwen_multimodal/FD001/seed42/`。除 token 外，每个 split
还保存 attention mask、RUL、发动机编号和窗口终点周期。完整性审计确认：所有
token 和标签均为有限值；train/val 分别覆盖 80/20 台训练发动机；test 覆盖全部
100 台测试发动机；标签范围符合 `[0, 125]` 截断规则。

## 主要文件

- `src/ts_mllm/qwen_multimodal.py`：动态 prompt、projector 和 token 选择；
- `src/ts_mllm/qwen_cache.py`：冻结模型加载、projector 训练、缓存和审计；
- `scripts/cache_qwen_multimodal.py`：命令行入口；
- `artifacts/qwen_multimodal/FD001/seed42/manifest.json`：可追溯配置和审计结果。

下一阶段可以直接读取这些缓存实现 TMAF，无需在每次融合训练时重复运行 Qwen。
