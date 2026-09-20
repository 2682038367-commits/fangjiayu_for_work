# FD001：B目标尺度下的完整TMAF

## 结果

训练及独立检查点重放均使用NVIDIA GeForce RTX 4060 Ti。固定split_seed42、
训练seed42，复用新stride50/DKE96/完整文本Qwen缓存与B时间基线验证最优权重。

| 配置 | 最优epoch | 验证RMSE | 测试RMSE | 测试MAE | 测试Score |
|---|---:|---:|---:|---:|---:|
| B时间基线 | 28 | 20.9830 | 21.5395 | 16.7660 | 1200.6823 |
| B完整TMAF | 28 | 16.8625 | 15.2218 | 11.0668 | 436.5904 |
| 论文FD001主结果 | — | — | 12.45 | 主表未给 | 233.40 |

以上实验指标遵循既有协议：预测恢复周期单位后裁剪到[0,125]计算；裁剪协议是
复现假设。TMAF未裁剪测试RMSE15.2344、MAE11.0772、Score437.2535。
相较B基线，验证RMSE降低4.1205（19.64%），测试RMSE降低6.3177（29.33%）。

## 配置与边界

- window40、样本stride50、Patch4/stride1/dim64/head1；batch128、lr0.002、30epochs。
- 时间分支与融合分支均lr0.002，冻结轮数0；不采用热身或分组小学习率。
- 保持原回归头、初始化与字面global_broadcast实现，不按结果更换token注意力。
- 训练309个窗口，每轮3次更新，共90次；验证76、测试100。
- B训练目标RUL/125，MSE；推理乘125，RUL上限仍为125。
- 原时间基线完成30轮训练，本次从其第28轮验证最优权重初始化，再训练TMAF30轮。
  因此不是等总训练预算的纯融合归因实验，不能仅由这张表证明Qwen贡献。
- B目标尺度、Adam/MSE、2层Transformer、平均池化、GELU、全局masked mean、
  预训练/冻结Qwen和MAE、时间分支初始化及验证选模等仍是已披露复现假设。
- 全局向量广播使Key相同、注意力均匀这一结构问题仍存在；本次不擅自更改。
- CUDA训练警告memory-efficient attention可能非确定性；仅一个seed，不宣称稳定性。

训练RMSE从139.7756降到13.9263，验证曲线有波动；按验证RMSE选择第28轮，
没有按测试指标选模。测试原始预测均值74.9007、标准差41.2408，并非常数。

## 核验与产物

独立GPU加载best.pt，验证与测试指标重现；检查点目标尺度、来源SHA256、
30轮学习率与零冻结、验证最优epoch、100条CSV预测/标签/发动机/周期及恢复单位
均通过。模块测试33项通过。旧产物未覆盖。

产物：`artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42/`
包含best.pt、history.json、result.json、output_audit.json、test_predictions.csv/SVG。

```bash
PYTHONPATH=src ../.venv/bin/python scripts/train_tmaf.py \
  --dataset FD001 \
  --rebuilt-data-dir artifacts/data_window40_stride50/split_seed42/FD001 \
  --cache-dir artifacts/qwen_audited/FD001/seed42 \
  --temporal-checkpoint artifacts/target_scale/FD001/seed42_verified/B_div125/best.pt \
  --output-dir artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42 \
  --train-stride 50 --validation-stride 50 --context-mode global_broadcast \
  --token-mode full --epochs 30 --batch-size 128 --learning-rate 0.002 \
  --temporal-learning-rate 0.002 --freeze-temporal-epochs 0 \
  --seed 42 --split-seed 42 --num-workers 2
PYTHONPATH=src ../.venv/bin/python scripts/audit_tmaf_scale_b.py
```

训练入口拒绝覆盖已有最优模型。下一步先核验论文明确的回归头，若一致不改；
再在固定B尺度下针对未说明的初始化做单变量验证。最终需要固定划分的多seed
实验及FD002–004，当前尚未达到论文性能或完成全部子集。
