# FD003 逐token注意力诊断（Fig. 12 prompt）

非论文公式的补充架构对照，主方案未替换。使用同一FD003 `figure12_v1` Qwen缓存、同一batch32时间权重、同一数据manifest、相同模型参数初始化SHA256和训练设置（seed42、30 epochs、batch32、Adam、两分支lr0.002、无冻结、RUL/125）。仅`context_mode=global_broadcast`改为`tokens`。GPU核验协议通过，见`FD003_TOKEN_ATTENTION_AUDIT.json`。

| 模型 | 验证RMSE | 最优轮次 | 测试RMSE | 测试MAE | 测试Score |
|---|---:|---:|---:|---:|---:|
| 纯时间batch32（参照） | 17.9442 | 22 | 19.4100 | 14.6750 | 1597.97 |
| Fig.12全局广播TMAF（论文公式字面版） | 17.4898 | 10 | **19.2494** | **14.1259** | **1626.79** |
| Fig.12逐Qwen token TMAF（非论文补充版） | **17.4149** | 10 | 20.4227 | 14.7152 | 1777.92 |

验证RMSE的0.075周期改善极小；固定模型测试RMSE反而高1.17周期，不能据此支持逐token架构。配置仅按验证集判断，不以测试结果调参。单种子不能说明普遍性。

## 注意力机制核验

全局广播版38个Key/Value相同，所有时间Query上的注意力严格均匀，各Key权重1/38，Q/K不起选择作用。

逐token版虽有513个缓存位置（约165个有效位置），却发生另一种退化。在**完整92个验证窗口，共3496个时间patch查询**中，最大注意力位置始终是缓存索引3；平均最大权重**0.9999992**，所有查询的最大权重均超过0.99；首末时间Query注意力分布的平均L1差仅**3.39e-7**。取8个验证窗口做反向传播探针，Q/K权重梯度范数约4.16e-8/8.96e-8，而Value梯度约5.10。掩码位置权重为0。说明“非均匀”不等于“有逐patch选择”。

索引0为视觉前缀；索引3按本地Qwen tokenizer对应固定标题`### Task Describe`中的` Describe`。Qwen3是因果模型，该位置的输出可以利用视觉前缀及此前固定标题，但不能直接看到后面的动态窗口统计（min/max/median/趋势）。因此逐token版**不等于没有任何样本信息**，但它没有检索后面那些动态文本token，也几乎不随时间Query变化。当前实验不足以判断是Qwen特征尺度、Key/Query投影初始化、softmax饱和还是上游对齐共同造成；不应贸然改温度、归一化或结构并称作论文复现。

新模型：`artifacts/fd003_figure12_v1/seed42/tmaf_tokens_batch32`。机制检查脚本：`scripts/audit_fd003_token_attention.py`。全局广播固定主结果与旧`legacy`文本A均保留。

补充训练前后数值定位：见`FIGURE12_CROSS_SUBSET_REPORT.md`及`FD003_TOKEN_COLLAPSE_DIAGNOSIS.json`。初始注意力是分散的，训练后才塌缩到索引3；验证缓存掩码及padding正确。因此不能把塌缩直接归咎于缓存错误。
