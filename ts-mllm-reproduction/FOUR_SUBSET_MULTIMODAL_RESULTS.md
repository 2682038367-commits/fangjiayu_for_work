# FD001–FD004完整多模态单种子结果

## 结论

FD002–FD004已逐集完成独立MAE预训练、DKE/线性视觉projector对齐、新频谱和完整
Qwen缓存、B时间基线、完整字面TMAF训练及独立GPU核验。FD001复用之前B时间基线
和默认偏置TMAF结果。主流程四集覆盖完成，**论文性能尚未复现成功**。

按用户决定停止输出偏置调优，不做稳定性实验；后续正式流程保留默认偏置，
统一目标尺度B=RUL/125。均值偏置的探索性结果仍完整保留，不作为本表主配置。

## 测试结果

| 子集 | B时间RMSE | 完整TMAF RMSE | TMAF MAE | TMAF Score | 论文RMSE | 论文Score |
|---|---:|---:|---:|---:|---:|---:|
| FD001 | 21.5395 | 15.2218 | 11.0668 | 436.5904 | 12.45 | 233.40 |
| FD002 | 43.9029 | 44.3397 | 37.5515 | 111609.3998 | 14.22 | 929.81 |
| FD003 | 34.6707 | 23.4251 | 18.1068 | 2976.4951 | 11.97 | 338.30 |
| FD004 | 44.7718 | 44.9046 | 37.8309 | 149654.7801 | 15.94 | 1715.11 |

论文值依据本地PDF第8页表III。当前主指标按既有假设：标签cap125、恢复周期单位、
预测clip[0,125]后计算；同时保存raw结果。FD002–FD004本轮最优检查点的测试raw
与clip指标一致。FD001 TMAF raw RMSE15.2344。MAE为补充指标，不伪称论文主表给出。

| 子集 | 时间最优epoch | 时间Val RMSE | TMAF最优epoch | TMAF Val RMSE |
|---|---:|---:|---:|---:|
| FD001 | 28 | 20.9830 | 28 | 16.8625 |
| FD002 | 30 | 42.0166 | 25 | 41.5967 |
| FD003 | 30 | 34.2462 | 30 | 19.0727 |
| FD004 | 28 | 41.3771 | 29 | 40.5794 |

两阶段均按验证RMSE保存最优，不按测试选epoch。TMAF是在验证最优时间权重基础上
再训练30轮，并非与单模态等总训练预算，不能单由提升表证明Qwen因果贡献。

## 固定协议与复现假设

- window40、样本stride50、Patch4/stride1/dim64/head1、14个有效传感器、Min-Max、RULcap125。
- split_seed42、模型seed42，发动机80/20划分；训练集拟合scaler，验证stride50、
  测试每发动机最后窗口及短轨迹左侧复制填充均为未明确协议的披露假设。
- 时间和TMAF各30轮、batch128、lr0.002；不冻结时间分支、不使用分组小学习率。
- 2层Transformer/FFN256/pre-norm/GELU/时间dropout0.1、平均池化及Adam/MSE仍为假设。
- 完整TMAF key64/output32、MLP512/dropout0.5；global_broadcast遵守字面描述，
  masked mean聚合、64维融合输出和具体回归头是解释假设；没有改为token注意力。
- Qwen3-0.6B冻结BF16，线性128→1024视觉projector，DKE96+位置→线性桥接→Qwen，
  文本上限512，保留全部有效文本输出和1个视觉前缀，缓存容量513、float16。
- MAE/对齐冻结策略、Qwen主型号、统计prompt、谱图算子细节、对齐目标与日程仍未
  获得作者唯一实现，不能把本流程称为已验证的完全论文方法。
- MAE独立预训练15轮、AdamW lr0.001、weight_decay0.05、mask0.75、小型patch19/
  embed128/2层/4头，是未公开的预训练假设；15轮不是替换论文最终训练30轮。
- 新MAE用训练stride50窗口在线GPU生成float32 RP/STFT/CWT，sensor logits固定均匀；
  随后对齐30轮，MAE/Qwen冻结，sensor fusion、文本桥接及视觉projector可学习。
- 对齐损失为冻结Qwen token embedding MSE与视觉prefix→masked teacher mean MSE，
  等权相加、训练词表SVD初始化，不使用RUL标签训练对齐目标；不证明获得RUL语义推理。

重要比较边界：FD001仍复用先前以legacy stride1频谱预训练的小MAE；FD002–FD004
则使用新stride50数据独立预训练。最终RUL数据/窗口/训练参数一致，但四集MAE
预训练来源协议尚不完全一致。该差异不是隐藏调优，不把四集称为完整统一预训练复现。

## 验收与问题

| 子集 | train/val/test | 每阶段更新数 | 时间测试raw std | TMAF测试raw std |
|---|---|---:|---:|---:|
| FD001 | 309/76/100 | 90 | 36.1899 | 41.2408 |
| FD002 | 812/195/259 | 210 | 0.7831 | 5.1326 |
| FD003 | 377/92/100 | 90 | 9.7531 | 36.0830 |
| FD004 | 938/220/248 | 240 | 0.3573 | 5.3252 |

所有测试预测条数正确且有限，时间基线GPU独立重放与CSV/标签/unit/cycle对应通过；
完整TMAF各自GPU最优权重重放与验证/测试指标、CSV恢复单位、最佳epoch、
30轮学习率和零冻结检查通过。MAE→对齐→缓存→时间→TMAF来源SHA256一致，
缓存dtype、shape、mask、padding、行映射、完整文本长度和文件checksum通过。

这仅表示数据与工程核验通过，不表示模型质量通过。FD002、FD004时间分支几乎
为常数，融合后变化仍很小；FD003时间分支预测方差也偏低。B尺度并非普遍解决方案。
global广播相同Key导致均匀注意力的结构限制仍然存在，本次不私自修改论文明文。
多工况与上述退化同时出现，是值得排查的线索，但尚未证明工况处理是原因。

未运行3seed稳定性，CUDA memory-efficient attention backward有非确定性警告；
不能宣称均值/标准差或稳定性达标。严格输入模态消融、few-shot与论文图3全寿命曲线
仍未完成；现有SVG是官方测试发动机终点预测折线，并非全寿命轨迹。

## 产物

- MAE：`artifacts/mae_pretrain_audited/{FD002–FD004}/stride50_seed42/`
- 对齐：`artifacts/svlma_alignment/{FD002–FD004}/stride50_seed42/`
- Qwen：`artifacts/qwen_audited/{FD002–FD004}/seed42/`
- 浮点频谱：`artifacts/spectrum_audited/{FD002–FD004}/stride50_seed42/`
- B时间：`artifacts/temporal_audited/{FD002–FD004}/stride50_B_seed42/`
- 完整TMAF：`artifacts/tmaf_audited/{FD001–FD004}/global_broadcast_stride50_B_seed42/`
- 汇总、过程日志和进度：`artifacts/multimodal_remaining_B_seed42/`
  包含summary.json、progress.json、four_subset_audit.json和每集每阶段log。

MAE、时间和TMAF保存验证最优best.pt、history/result；TMAF另存initialization、
output_audit、测试CSV/SVG。已有模型未覆盖。FD002 MAE报告中未使用的legacy cache_dir
占位字段已校正为null并记录实际rebuilt_data_dir，未改变训练、权重或指标。

```bash
PYTHONPATH=src ../.venv/bin/python scripts/run_remaining_multimodal.py
PYTHONPATH=src ../.venv/bin/python scripts/audit_four_subset_results.py
```

批量入口支持`--datasets FD002`等子集选择，复用已完成且来源相符的阶段；部分失败
不自动覆盖已有模型，查看对应日志处理，不回退CPU。模块测试41项通过，compileall通过。

## 下一步优先级

先诊断FD002/FD004的时间分支退化：核查多工况下归一化、输入/标签统计、训练梯度
与预测随标签变化，而不是叠加融合调优。论文明确的参数保持；按工况归一化等
若论文未规定，只能作为单独披露的候选假设，不擅自替换主协议。
随后如需严格统一四集MAE来源，独立重建FD001的stride50 MAE及下游产物，保留本表历史结果。
