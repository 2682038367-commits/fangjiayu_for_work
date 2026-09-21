# FD003 输入级模态消融（seed 42）

本报告对应论文 Fig. 6 的 `w/o Visual`、`w/o Textual` 概念，但只报告一次固定划分/随机种子的复现结果。模型为论文公式版 `global_broadcast` TMAF；未用逐 token 诊断版。

| Qwen 输入 | 验证 RMSE | 验证 MAE | 验证 Score | 测试 RMSE | 测试 MAE | 测试 Score |
|---|---:|---:|---:|---:|---:|---:|
| full (temporal + visual + textual) | 17.490 | 11.838 | 752.0 | 19.249 | 14.126 | 1626.8 |
| w/o Visual (temporal + textual) | 17.505 | 10.584 | 1164.4 | 22.881 | 16.063 | 3674.9 |
| w/o Textual (temporal + visual) | 17.288 | 12.382 | 755.6 | 23.057 | 17.188 | 2454.9 |

## 审计结论

- 三组的重建数据哈希、时间分支权重哈希、batch=32、seed/split=42、样本步长=50、Figure12 文本配置及 TMAF 超参数完全一致；唯一实验变量是写入 Qwen 的输入模态。
- `w/o Visual` 的 Qwen 输入仅为 DKE 文本 embedding：没有生成频谱、没有调用视觉编码器、没有视觉前缀。
- `w/o Textual` 的 Qwen 输入仅为投影后的视觉前缀：未生成或分词 prompt，所有样本有效 token 数恰为 1。
- 这不是在 Qwen 输出之后屏蔽 token，因此文本 token 不可能先读取视觉前缀后再被遮蔽。

## 解释边界

按验证 RMSE，去视觉较完整模型差 0.015，近似持平；去文本反而好 0.202。因此这一次运行没有支持文本或视觉提供稳定、独立增益的结论。测试集数值仅作保留的泛化观察，不能用于选择模型。

视觉 projector 的上游对齐训练曾以文本 teacher embedding 为目标，这是论文未公开的复现假设；故 `w/o Textual` 严格表示 Qwen 推理输入没有文本，而不表示视觉 projector 的训练过程没有任何文本监督。