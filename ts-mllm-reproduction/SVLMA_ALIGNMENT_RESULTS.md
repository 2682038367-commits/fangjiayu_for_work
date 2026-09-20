# 优先级2/3：96维DKE实际接入与线性视觉projector

FD001实施完成，使用独立stride50数据（train309/val76/test100）在RTX4060Ti上运行。
本阶段不训练RUL回归器，不使用RUL标签进行适配对齐，不把对齐MSE当作论文RUL指标。

## 实际输入路径

```text
文本token IDs → Embedding(151936,96) + Position(512,96)
             → Linear(96,1024, bias=False) ────────────────┐
                                                        ├→ Concat → Qwen inputs_embeds
40×14 → RP/STFT/CWT → 预训练MAE128 → Linear(128,1024) ──┘
```

独立96维DKE不再只是一个未使用模块。`compose_multimodal_embeddings`被训练后Qwen
GPU验证和缓存写入函数共同使用，且测试禁止其绕过DKE直接使用Qwen原生embedding。
视觉projector只有单个Linear，无GELU、额外Linear或LayerNorm，也未加载旧MLP权重。
最大文本512；拼接后最多513个输入token，512是否包含视觉是按文本长度解释的假设。

## 明确设置与补充假设

保持论文明确参数：文本96、视觉128、文本上限512、视觉线性映射；本阶段统一lr0.002、
batch128、30epochs，不引入分组小学习率。窗口40/sample stride50/patch stride1保持不变。
论文未公开是否存在独立对齐阶段，因此30轮适配训练不能说成论文明确的原始阶段日程。

本次固定未公开部分，不作隐性调参：

- A-DKE-01：96→Qwen1024使用无偏置线性桥接；
- A-DKE-02：从训练prompt出现的token对应Qwen原生embedding做SVD初始化，
  再将预训练词表投影到96维；词表不足96时用正交补空间补足。初始化只使用训练
  prompt来选择基，不用val/test标签；可学习位置参数从零开始训练；
- A-ALIGN-01：文本桥接用冻结Qwen原生token embedding的masked MSE做蒸馏；
- A-ALIGN-02：视觉前缀对齐同一prompt原生embedding的masked mean，MSE与文本loss等权；
- A-ALIGN-03：冻结MAE/Qwen，Adam训练DKE、桥接、线性projector及频谱通道融合；
- 频谱两组softmax融合权重从均匀logits重新初始化，在新stride50窗口在线学习。
  不加载旧CNN融合权重，不复用旧uint8谱图或Qwen缓存；
- 使用现有本地小MAE预训练encoder与Qwen3-0.6B BF16，权重来源/型号/骨干冻结、
  prompt模板和时频具体参数仍属于已披露假设。

初始化及loss选择均不是论文明文。零初始化位置参数与原生词嵌入蒸馏可能弱化显式
位置信号；不能将它或非常低的文本重建MSE解读为已证明语义/位置建模有效。
训练词表较小且96维容量足够，文本重建极低也不能证明专家知识推理或RUL提升。

## FD001执行结果

- 最优对齐checkpoint：epoch30（按val文本MSE+视觉MSE）；
- 验证文本MSE：约3.01e-11；
- 验证视觉MSE：0.0003668502（第1轮约0.072752）；
- 真实桥接输入的Qwen前向：val76/test100条，全部有限，输出各样本[513,1024]；
- 训练后的频谱融合logits非零；线性projector/DKE/位置/桥接均有梯度路径测试；
- 单元测试28项通过。

最终使用 `artifacts/svlma_alignment/FD001/stride50_seed42/best.pt`，包含DKE、bridge、
线性projector和新频谱融合state，以及数据manifest SHA256和MAE来源。
另有history.json/result.json。早先`.../seed42/`为继承旧CNN频谱权重的pilot，
保留但不再使用；默认入口明确指向stride50_seed42产物。

## 入口及后续

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_svlma_alignment.py
```

训练脚本拒绝覆盖已有目录，复跑需指定新的--output-dir。实现目前实际运行FD001，
尚未为FD002–004训练对应适配权重。

`scripts/cache_qwen_multimodal.py`默认读取新数据及训练后的DKE/线性projector/频谱state，
在线生成float32频谱，并检查数据manifest和MAE来源；不再默认走原生文本embedding。
本次**尚未写出全量Qwen token缓存，也没有新的TMAF/RUL指标**。下一阶段可生成完整
token缓存，再按论文明确的统一学习率训练字面TMAF。现有旧RUL指标不可套用到此路径。
