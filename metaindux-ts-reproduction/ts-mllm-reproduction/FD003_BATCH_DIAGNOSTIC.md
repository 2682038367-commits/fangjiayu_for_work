# FD003 batch size 单变量诊断

GPU完成batch128及batch32独立训练。保存于 `artifacts/fd003_batch_diagnostic/seed42/batch128` 与 `batch32`，旧时间权重及主TMAF未替换。

固定：seed42、split_seed42、相同global Min-Max数据manifest、30epochs、Adam lr0.002、RUL/125、窗口40、样本/验证步长50、Patch4步长1、原时间网络。仅batch size改变。两组初始权重SHA256均为 `b95ee0d0cd7360e8a9046f2280ba03b93eb84bbfa3242b565f1c14dd68d6bc64`。其他train_config字段及model_config一致。batch128重放的曲线与原基线逐轮相符。

| 指标 | batch128 | batch32 |
|---|---:|---:|
| 每轮/总更新次数 | 3/90 | 12/360 |
| 最优epoch（按验证RMSE） | 30 | 22 |
| 验证RMSE | 34.2462 | 17.9442 |
| 验证预测std | 10.1536 | 36.7106 |
| 测试RMSE（截断） | 34.6707 | 19.4100 |
| 测试MAE（截断） | 29.1833 | 14.6750 |
| 测试Score（截断） | 9004.45 | 1597.97 |
| 测试预测std（截断） | 9.7531 | 36.6892 |
| 测试原始RMSE | 34.6707 | 20.2152 |
| 测试数量 | 100 | 100 |

配置优劣依据验证RMSE，不用测试选择；测试仅作固定对照的补充报告。改善支持旧基线明显训练不足，batch size是重要训练因素；由于同时影响更新次数及梯度噪声，不能严格归因于更新次数本身。单seed不代表稳定性结论。

batch32后期验证仍有振荡，不保证已充分收敛或与论文达到相同水平。30轮不变。batch32是论文未明确的复现假设，不称为论文公开方法。

新时间基线测试RMSE19.41低于旧主TMAF23.43，但旧TMAF从较弱batch128时间权重出发、使用batch128；不能用这两个不同训练条件的产物判断融合增益。后续应从新时间权重初始化TMAF，并统一batch32，保留原promptA及其他配置，作为独立诊断版本；prompt修复另行进行，避免混杂。
