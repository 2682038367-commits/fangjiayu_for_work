# TMAF 方程核验与 Fig. 12 prompt 修复（FD003）

## 方程逐项对应

论文 III-D Eq.14–19：时间patch特征为Query；全局视觉语言向量分别线性投影为Key、Value，广播到时间patch数；缩放点积注意力、Concat时间与检索上下文、线性融合，最后回归。当前 `src/ts_mllm/tmaf.py` 符合该**字面计算顺序**，Key维64、Value维32、时间token数38。全局向量的masked mean提取、最终时间mean pooling、512-GELU-Dropout回归头及训练策略是论文未充分说明的复现假设，不能称为作者原实现。Eq.18/19的线性Concat融合确实保留正负权重，但不是固定恒等残差。

**关键数学缺陷**：全局向量广播后38个Key完全相同，38个Value也完全相同。每个Query对所有Key的logit相同，所以softmax必为1/38；每个patch得到相同Value。融合后再对patch取均值时，网络退化为“平均时间表示＋全局视觉语言表示”的后期融合。时间Query和语义Key不会学习任何有用的逐patch选择关系。这既是当前实现的事实，也是论文公式的字面退化；论文正文声称的“对局部行为选择语义细节”无法从这个字面公式推出。不能因此私自把正式论文复现改成token-level交叉注意力；那将是另一架构假设。

GPU对已训练FD003 batch32 TMAF核验（`scripts/audit_tmaf_broadcast.py`）：注意力偏离1/38的最大值0；fused与直接加性计算最大差5.36e-7，最终预测差1.19e-7。Q/K投影权重梯度约2e-6（浮点数值残差），Value/Fusion梯度分别19.79/14.06。审计输出 `FD003_TMAF_BROADCAST_AUDIT.json`。

## Fig. 12 prompt 修复

新独立`figure12_v1` profile，保留旧`legacy`文本A及所有主产物。三个块：Task Describe（航空发动机/C-MAPSS/RUL任务），Dataset Feature（当前40×14窗口的min、max、median、整体趋势，并指示结合前面的视觉特征），Dataset Describe（数据集和输入说明）。没有伪造具体故障机理或逐周期运行设置；原始数据有运行设置，但本profile未输入。

**复现假设**：min/max/median对40×14全部归一化数值聚合；趋势由末10周期整体均值减首10周期整体均值，阈值±0.03；具体措辞不是论文发布的完整prompt。新模板见`src/ts_mllm/figure12_prompt.py`。所有FD003 569个样本token长度164–165（视觉前缀另占1个），低于512，不发生文本截断。

沿用同一FD003 batch32时间权重，独立重跑对齐30轮、Qwen缓存及TMAF30轮；MAE、数据和时间基线未重跑。新缓存train/val/test数量377/92/100，标签和发动机映射核验通过。TMAF仍采用global_broadcast/mean，故注意力退化仍存在。

| FD003、seed42、batch32 | 验证RMSE | 测试RMSE | 测试MAE | 测试Score |
|---|---:|---:|---:|---:|
| 纯时间 | 17.9442 | 19.4100 | 14.6750 | 1597.97 |
| 旧文本A TMAF | 17.7398 | 22.0753 | 16.2598 | 2505.80 |
| Fig.12 prompt管线 TMAF | **17.4898** | **19.2494** | **14.1259** | 1626.79 |

新prompt管线相对旧A改善验证和测试RMSE，但测试RMSE仅比纯时间低0.16周期，Score略差；**不能声称多模态有稳健增益**。更换prompt需要重训对齐并重建Qwen缓存，且旧/新对齐各自学习了频谱融合参数，因此这是整条prompt管线对照，并非只替换文本措辞的严格因果消融。仅一个种子，测试指标不用于选配置。

新产物：`artifacts/fd003_figure12_v1/seed42/{alignment,qwen,spectrum,tmaf_batch32}`。固定主方案仍为旧文本A。若需跨FD001–FD004推广，应先决定如何处理退化的字面TMAF，并分别标注“论文公式复现”与“非原文token级修正”的架构地位。
