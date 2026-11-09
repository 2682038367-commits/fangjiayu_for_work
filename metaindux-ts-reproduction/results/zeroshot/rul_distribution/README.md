# RUL分布统计（2026-10-22）

[Observed] 统计本次零样本w48实验FD001–FD004全部5个生成种子的NPZ标签，以及实际预处理后的真实训练窗口标签。逐种子检查：20/20份生成数据的RUL标签与对应真实训练窗口标签逐项完全相同。因此每种子的计数分布完全一致，合并5种子的比例也一致、计数为真实训练窗口的5倍。

[Code] RUL标签由窗口末端定义，超过125截断到125；生成时直接以真实训练标签作为条件并原样保存，不是由生成传感器信号测出来的寿命。相同标签分布不构成生成质量或退化物理一致性的证据。零样本生成器训练只使用RUL>30窗口，但采样条件覆盖完整标签范围。

[Observed] 来源口径：real_train_windows为完整真实训练窗口，generator_training_subset为实际生成器训练子集，generated_seed3为单个生成种子，generated_all_5_seeds为合并5种子，real_test_terminal为每台测试发动机的末端RUL，real_test_fatigue_evaluation为正式评估所用RUL<=30测试子集。不能混淆真实训练分布与实际零样本生成器训练分布；后者没有RUL<=30数据。

[Code] RUL>30是论文的正常阶段定义，不是经过验证的未退化健康阶段；RUL=125包含所有被截断到125的更大寿命值，并非原始寿命都恰好125。

产物：real_vs_generated_rul.png/pdf，summary.csv，exact_rul_counts.csv（逐个整数RUL的数量与百分比），label_equality_audit.json。
复算：`.venv/bin/python analysis/diagnostics/rul_distribution.py`。

本次为已有产物的支持性描述分析，没有新训练、改变复现协议或重新判定论文指标。
