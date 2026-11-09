# 未截断RUL>125的窗口分布

[Observed] 来源为4个子集的真实训练发动机记录。先按每台发动机最大cycle减去当前cycle恢复原始RUL，再取48步窗口末端的值，步长1、排除engine1000。统计对象不是完整发动机数，也不是全部逐周期记录。对原始RUL>125筛选，bins.csv的百分比以该子集所有>125窗口为分母；原始RUL恰好125另列summary.csv。

[Observed] 重建的总窗口数及RUL>=125数量已与既有数据加载审计核对一致。原始RUL范围FD001126–314、FD002126–330、FD003126–477、FD004126–495。

[Code] 生成文件中的条件标签最大125，无法从生成信号中恢复一个已知的原始>125标签。真实源窗口原始RUL的分布不是生成窗口已验证的物理寿命分布；保存顺序的来源索引也不赋予其真实物理寿命。

产物：bins.csv、summary.csv、exact_raw_rul_counts.csv、uncapped_rul_distribution.png/pdf、audit.json。复算：`.venv/bin/python analysis/diagnostics/uncapped_rul_tail.py`。只读描述分析，无新训练或严格论文结果判定。
