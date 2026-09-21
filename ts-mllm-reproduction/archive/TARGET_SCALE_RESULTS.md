# 目标尺度单变量验证：原始RUL vs RUL/125

FD001、RTX4060Ti，使用同一stride50导出数据（309/76/100）、split seed42、model seed42。
两组初始化权重SHA256完全相同，训练DataLoader均使用同一独立seed42生成器；初始诊断
使用额外非shuffle DataLoader，不消耗训练随机顺序。回归头均为64→1，Transformer配置
不变，Adam/MSE、lr0.002、batch128、30epochs、每组90次更新。没有训练Qwen或TMAF。

唯一改变的工作变量是目标/输出单位：

- A_raw：训练target=RUL，模型输出周期单位；
- B_div125：训练target=RUL/125，模型输出归一化单位；评估和CSV一律乘125恢复周期。

RULcap仍125，未改标签定义或输入Min-Max。目标归一化是论文未明确的复现假设。
这不等于在固定物理输出下只给MSE乘一个常数：同一原生初始化输出在B中恢复为周期
后会乘125，物理输出尺度和有效更新尺度随参数化改变，不应把增益简单解释为梯度小。

## 1. 数据、梯度及输出读取检查

- 从原始文件独立重建全485个窗口，与导出x/target/unit/cycle逐行完全一致；
- 原始训练标签均值83.4660周期，测试标签均值74.4500周期；两组相同；
- 同一首批128样本的原始标签均值82.546875：A训练target均值82.546875，
  B为0.660375；两组原生初始预测均值都约-0.667229；
- 每步检查head、patch投影、位置参数、第一层attention与output norm梯度存在且有限；
- 最优checkpoint相对初始权重的head、patch、位置参数均有实际变化；
- GPU直接计算head(mean(encode(x)))与model(x)一致；保存CSV的raw预测与模型输出
  按target_divisor恢复的周期值一致，发动机与标签也一致。

| 首批梯度L2 | A_raw | B_div125 |
|---|---:|---:|
| regression_head.weight | 1267.1129 | 20.1857 |
| patch_projection.weight | 813.9822 | 12.9476 |
| 第1层attention in_proj | 558.0848 | 8.8533 |

梯度数值依赖损失单位。A没有NaN/Inf，也没有断梯度，不能仅凭大梯度宣称梯度爆炸。
head.weight的实际参数变化L2为A=1.3987/B=0.0889，说明两组Adam均确实更新参数。

## 2. 验证与测试结果

与既有协议一致，checkpoint选择使用原周期单位的验证clip[0,125] RMSE；先完成两组
训练、写selection.json选B，然后才读取测试预测。测试不用于选择尺度或最佳epoch。

| 配置 | 最优epoch | Val RMSE | Test RMSE | Test MAE | Test Score |
|---|---:|---:|---:|---:|---:|
| A_raw | 30 | 74.9039 | 66.2019 | 55.4415 | 63813.18 |
| B_div125 | 28 | **20.9830** | **21.5395** | **16.7660** | **1200.68** |

上表clip协议仍是未明确的评估假设，不是论文强制规定。未clip测试RMSE为A=66.2019、
B=22.1054，因此B改善不只是clip造成的。两组测试均100个有限预测。

| 最优模型测试原始预测统计 | A_raw | B_div125 | 真实标签 |
|---|---:|---:|---:|
| 均值，周期 | 21.7526 | 80.6948 | 74.4500 |
| 标准差，周期 | 0.0031 | 36.1899 | 40.0733 |

B原始预测范围约0.5363–144.1905，clip后标准差35.0585。A原始预测范围约
21.7432–21.7553，几乎常数；B输出对样本有明显变化。

## 3. 结论与下一步

数据对应、标签读取、反向传播连通性、实际参数更新与预测单位恢复未发现错误。
A在有限90次更新内几乎只学到常数输出；B显著改善验证集并摆脱这一退化，支持
“目标/输出数值尺度是当前时间基线训练困难的重要因素”，但并非唯一因素证明。

仅seed42；B仍明显未达到论文RMSE12.45。下一步回归头对照可以固定B目标尺度，
只改变未公开的头结构，不调整论文明确的batch/lr/epochs/dim。应随后做多seed验证。
正式训练默认值、旧模型和TMAF结果未改写，不把实验B自动当成论文原始方法。

## 文件与复跑

主产物：artifacts/target_scale/FD001/seed42_verified/。
包含A/B各自best.pt（记录target_divisor）、history.json、result.json、预测CSV/SVG；
以及summary.json、selection.json、output_audit.json。B checkpoint的原生输出不是RUL
周期，未来读取必须乘其target_divisor，不可直接调用旧无尺度推理函数解释输出。

```bash
PYTHONPATH=src ../.venv/bin/python scripts/compare_target_scale.py
PYTHONPATH=src ../.venv/bin/python scripts/audit_target_scale_outputs.py
```

训练入口拒绝覆盖已有目录，复跑指定新的--output-root。早先seed42目录为诊断消耗
shuffle顺序的pilot，保留但不用于本报告。单元测试32项通过。
