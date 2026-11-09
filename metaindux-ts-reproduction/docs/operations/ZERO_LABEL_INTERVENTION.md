# RUL=0标签混用单因素验证

2026-10-23，用户授权“优先验证0标签混用”。

- 范围：diagnostic / supporting，合约 `scope-20261023-zero-label-pair`；不是新的论文严格结果。
- Run ID：`diagnostic-zero-label-pair-fd001-seed3-20261023`。
- 对照：在当前环境重新训练FD001 full-data w48、seed3，batch标签置零概率0.1。
- 干预：同配置、同种子，只有batch标签置零概率变为0。
- 保留：网络已有null embedding与条件dropout、频率机制、初始化种子、Adam、学习率调度、70epoch、batch256、梯度裁剪、1000步linear DDPM、完整采样标签、best-loss权重选择、同一组5个真实数据预测器。
- 即使概率为0，也消耗原有一次NumPy随机抽样，避免因跳过抽样额外扰动后续随机数序列。
- 原默认概率仍是0.1；本次没有把所有历史配置默认改为修复版。独立产物后缀为`zero_label_control` / `zero_label_no_batch_zero`。

预先判据：原版复现RUL0平均预测>30；修复版的RUL0平均预测与RMSE均降至原版的50%以下，则支持本次单种子因果解释。若原版未复现异常，则判据不能成立。50%是本地诊断阈值，不是论文标准。报告RUL1及其他段的变化，不因结果调整参数。单种子/代理预测器不能证明物理真实性或所有数据集的下游改善。

预算：仅2次生成器训练与完整采样，之后自动用5个固定真实预测器评估全部生成窗口，结束后不自动扩展种子、数据集或下游重训。失败即停止，运行/失败/结果写入快照与ledger。

配置：`configs/diagnostic_zero_label_fd001_{control,no_batch_zero}.yaml`。
监督进程与原始快照：`runs/diagnostic-zero-label-pair-fd001-seed3-20261023/`。
后台服务：`metaindux-zero-label-pair-20261023.service`。
日志：`logs/zero_label_pair_fd001_seed3_20261023_console.log`。
评估：`analysis/diagnostics/evaluate_zero_label_pair.py`。
结果：`results/diagnostics/zero_label_pair_fd001_seed3/{comparison.csv,verdict.json,result.md}`。

当前阶段：已启动，最终指标待执行结束；不将计划判据当作已测结果。

## 完成记录（2026-10-23）

[Observed] 两组及自动评估完成，审计通过。RUL0平均预测66.855→1.427，RMSE79.412→2.623，预设判据通过。排除0后RMSE7.991→7.948，未测下游真实测试改善。完整分析：`docs/conclusions/final/zero_label_pair_fd001_seed3_report.md`。上文“已启动、待结束”为历史状态，已由本记录取代；无自动后续训练。

## FD002单种子验证（2026-10-23）

用户授权FD002种子3按相同设计验证：当前环境原版prob0.1与禁用版prob0，均70轮、batch256、完整DDPM采样。预设判据与FD001相同。GPU后台服务`metaindux-zero-label-pair-fd002-20261023.service`；快照`runs/diagnostic-zero-label-pair-fd002-seed3-20261023/`；自动结果`results/diagnostics/zero_label_pair_fd002_seed3/`；日志`logs/zero_label_pair_fd002_seed3_20261023_console.log`。结果待完成，不自动扩展种子或重训下游。
