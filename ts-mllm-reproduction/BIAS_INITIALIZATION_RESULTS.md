# FD001输出偏置初始化单变量对照

## 结果与选择

全部RUL训练目标仍为用户统一的B尺度RUL/125，预测乘125恢复周期单位。
本报告A/B仅指偏置初始化选项，不是原始RUL与归一化RUL对照。

| 偏置选项 | 最优epoch | 验证RMSE | 测试RMSE | MAE | Score |
|---|---:|---:|---:|---:|---:|
| A：默认bias | 28 | 16.8625 | 15.2218 | 11.0668 | 436.5904 |
| B：训练标签均值/125 | 15 | 16.5992 | 18.0168 | 12.7174 | 961.9117 |

按预定验证RMSE标准，均值bias胜出：降低0.2633（约1.56%）。但测试RMSE变差
2.7951，MAE与Score也变差。完整保留结果，不因测试结果反向选默认bias，也不
声称均值bias改善泛化或解决初始化问题。单seed、小验证集和CUDA非确定性使这点
验证优势不足以证明稳定性。CLI默认bias仍为default，实验选项未自动变成正式方法。

验证与测试指标在恢复单位后裁剪到[0,125]计算，这是既有披露假设；均值bias
检查点的测试预测本身在范围内，raw指标与裁剪指标相同。

## 固定项与唯一改动

- 固定split_seed42、seed42、训练309/验证76/测试100。
- window40/sample-stride50、Patch4/stride1/dim64/head1；完整字面global_broadcast TMAF。
- 同一B时间基线第28轮验证最优权重和同一新Qwen完整文本缓存；来源SHA核验一致。
- 不改回归头、激活、pooling、其他层初值；仅修改regression_head.3.bias。
- 默认bias=-0.0130560771；训练309窗口标签均值83.4660194，均值bias=0.6677281260。
  均值仅来自训练标签，不读取验证或测试标签设置偏置。
- Adam/MSE、batch128、两组lr均0.002、冻结0轮、30轮90次更新；按验证RMSE保存。
- 均值偏置是论文未明确的复现假设，不是论文方法。不修改既有B目标尺度默认。

CPU统计训练均值及初始化审计不消费随机训练loader，不改变shuffle或dropout随机流。
重新构造seed42默认TMAF并加载同一时间权重，逐张量核验改动只有最后bias。
除bias外初态SHA256均为：
`e700e183c4f6597b58fd5a256a89131b697e4d13bd37da4f9020df0fac37b6ff`。
与实际训练记录的修改前/后初态SHA一致。

局限：历史A没有保存完整初态，因此A初态是由未变的构造/加载路径及seed重建，
并核对上轮独立审计中默认bias；不能把这描述为逐张量读取A历史初态文件。
PyTorch训练仍警告memory-efficient attention backward可能非确定性。

## 验证

GPU为NVIDIA GeForce RTX 4060 Ti。均值bias训练RMSE91.0121→11.8443，但验证曲线
后期有波动，第15轮验证最优；最终训练误差较低不代表验证更好。
独立GPU重放第15轮最优权重，验证与测试指标重现；100条测试结果有限、非恒定，
CSV预测/标签/unit/cycle与模型输出一致；原始预测均值80.0494、std40.9636。
初始化配对、数据/缓存/时间权重来源、30轮统一学习率与零冻结检查均通过。
模块测试37项通过；旧模型和结果未覆盖。

## 产物与运行

新产物`artifacts/tmaf_bias_init/FD001/train_mean_seed42/`含best.pt、initialization.json、
selection.json、history.json、result.json、output_audit.json及测试CSV/SVG。
旧A仍位于`artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42/`。

使用此前完整TMAF命令，另设独立输出目录并追加`--output-bias-init train_mean`；
不传该参数则保持默认bias。核验命令：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/audit_bias_initialization.py
```

## 下一步

暂不继续叠加调优。固定划分，补seed52/62的两种bias配对对照，汇总3seed验证及
测试RMSE/MAE/Score均值和样本标准差，按验证均值比较，再固定初始化配置。
若复用同一B时间checkpoint，仅验证融合阶段随机性；若评估完整流程稳定性，
需要每个seed各自训练B时间分支再训练融合，不能混淆两种稳定性定义。
