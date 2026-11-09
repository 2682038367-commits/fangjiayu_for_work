# 固定主方案 main_protocol_v1

固定选择与来源hash记录在[configs/main_protocol_v1.json](configs/main_protocol_v1.json)。
这是一份主结果选择清单，不会修改或重跑旧实验，不代表严格忠实的作者实现。
后续报告使用下列结果，不能混用新prompt、末token提取或visual_only消融指标。

| 子集 | 数据归一化 | 文本 | Test RMSE | MAE | Score | 输出数量 |
|---|---|---|---:|---:|---:|---:|
| FD001 | 全局Min-Max | 原统计A | 16.3233 | 11.8122 | 575.46 | 100 |
| FD002 | 按工况Min-Max | 原统计A | 17.0862 | 12.1934 | 1551.20 | 259 |
| FD003 | 全局Min-Max | 原统计A | 23.4251 | 18.1068 | 2976.50 | 100 |
| FD004 | 按工况Min-Max | 原统计A | 18.9958 | 13.1156 | 2465.36 | 248 |

主表沿用clip[0,125]评估，raw指标单独保存在清单；clip本身为评估假设。
工况Min-Max是文献支持的补充预处理，不是TS-MLLM明确报告的原始方法。

## FD001修复分析

| 指标 | 旧MAE链路 | 统一stride50 MAE链路 |
|---|---:|---:|
| 验证RMSE | 16.8625 | 17.0594 |
| 测试RMSE | 15.2218 | 16.3233 |
| 测试MAE | 11.0668 | 11.8122 |
| 测试Score | 436.59 | 575.46 |

测试RMSE增加1.1015周期（7.24%）。原始预测标准差41.12、与标签相关系数0.921，
未出现近似常数。训练RMSE从158.39降到13.34，验证最优第18轮，末轮验证17.73。
统一MAE训练样本协议、谱图精度和初始化融合来源并不保证性能改善；
为统一来源，主结果采用修复后版本，不退回测试指标更好的旧MAE版本。

修复脚本原本假定旧FD001结果包含初始化hash，实际缺少该字段，导致训练结束后
比较检查失败、没有生成output_audit。现已修正，未重新训练，GPU独立回放通过，
76个验证/100个测试样本及CSV正确。旧新初始化是否逐参数相同不可核验；
相同seed不等同于已证明相同初始化，不能称为严格配对单变量实验。
FD003来源已通过核验，无重跑。

## 不变设置及剩余假设

四集均window40、sample stride50、Patch4/stride1、dim64/head1、2层Transformer；
MAE均采用stride50窗口的在线float32/均匀初始融合预训练流程；原统计文本、
DKE96、完整512文本缓存、global_broadcast+mean、full token。
TMAF30轮、batch128、Adam统一lr0.002、无冻结、默认初始化、seed/split42；
训练目标RUL/125，验证选最优。模型结构和训练参数未为固定方案做调整。
2层、MAE及桥接/对齐目标、mean池化、目标尺度等未说明设置继续标为复现假设。
广播重复Key/Value的均匀注意力限制保留，未擅自改为token交叉注意力。
单seed，未证明稳定性；FD003仍明显落后，不能再认为全部剩余误差只来自多工况。

检查锁定文件是否变化：

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/check_main_protocol.py
```

该命令仅做只读文件hash/配置核验，不训练、不改缓存。
