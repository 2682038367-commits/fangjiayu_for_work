# FD004：相同工况归一化协议的时间基线验证

## 结果

在RTX4060Ti上训练及独立重放。严格复用FD002协议：三个setting训练集标准化、
KMeans6(seed42/n_init10)、每工况每传感器Min-Max。两组均采用目标RUL/125。

| 归一化 | Val RMSE | Test RMSE | Test MAE | Test Score | Test raw预测std | Test raw预测/标签相关系数 |
|---|---:|---:|---:|---:|---:|---:|
| A：全局Min-Max | 41.3771 | 44.7718 | 37.8260 | 128692.4015 | 0.3573 | 0.1374 |
| B：按工况Min-Max | 19.0214 | 20.7756 | 16.0984 | 2961.4310 | 39.7345 | 0.8737 |

按验证RMSE选B。A验证最优epoch28，B为30；两组都训练30轮，每阶段240次更新。
主指标先恢复周期单位再clip[0,125]；B raw测试RMSE21.2911、MAE16.9381、
Score3066.8656，clip预测std38.3900。标签cap和预测裁剪是既有披露假设。

结果支持归一化选择是当前多工况时间基线退化的重要因素：不仅预测方差恢复，
预测与标签相关性也明显增加。未证明唯一原因；单seed不证明稳定泛化。
这仍是时间基线，不能当成完整TS-MLLM与论文FD004 RMSE15.94直接等模型比较。

## 固定协议及配对核验

- 保留原split_seed42划分，199训练发动机、50验证发动机；窗口938/220/248。
- scaler的setting均值/标准差、六个聚类中心、sensor min/max全部只在训练发动机拟合；
  验证和测试使用最近训练中心，没有重新拟合，没有传感器输入clip。
- 设置仅用于工况识别，不新增模型输入维度，仍为14个传感器。
- window40/sample-stride50、Patch4/stride1/dim64/head1、RULcap125，目标RUL/125。
- 2层Transformer、平均池化Linear时间头、Adam/MSE、batch128、lr0.002、30轮，
  seed42/split_seed42/CPU线程4，不调其他初值或参数。
- 两组实际记录的初态SHA256相同：
  `b95ee0d0cd7360e8a9046f2280ba03b93eb84bbfa3242b565f1c14dd68d6bc64`。
- 为保持FD002同一配对证据，本次重跑A记录初态；精确复现原全局归一化结果，
  不是增加训练预算或多seed稳定性实验。

六个setting中心约为(0,0,100)、(10,0.25,100)、(20,0.70,100)、(25,0.62,60)、
(35,0.84,100)、(42,0.84,100)。训练归一化后同工况传感器std中位数0.1431。
验证/测试传感器元素超界比例约0.0396%/0.0106%，保留而不裁剪。

论文明确Min-Max，但没有明确按工况拟合；条件化拟合、KMeans6及识别细节是
复现假设，不称为作者已确认方法。原单模态头、optimizer等未说明部分也保留假设标注。

## 验证及产物

原始文件SHA与原全局数据一致；独立训练集重拟合scaler与保存参数匹配，
全部938/220/248窗口x/target/unit/cycle从原始文件重建一致；target/unit/cycle/
start_row/left_padding与A逐元素相同。NASA RUL文件对应的248台发动机末窗口检查通过。
GPU重放A/B验证及测试指标、CSV周期单位和行映射正确；scaler文件与对应数据源
一致。44项测试及compileall通过。CUDA backward仍有非确定性警告，未做3seed。

- 数据：`artifacts/data_condition_minmax/split_seed42/FD004/`
- B模型：`artifacts/temporal_condition_minmax/FD004/stride50_B_seed42/`
- A配对模型：`artifacts/normalization_comparison/FD004/global_minmax_seed42/`

B含best.pt、history/result、selection/output_audit、真实scaler、测试CSV/SVG。
旧数据、模型与频谱/Qwen缓存均未覆盖。

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src ../.venv/bin/python scripts/rebuild_condition_data.py --dataset FD004
PYTHONPATH=src ../.venv/bin/python scripts/train_temporal.py \
  --dataset FD004 --rebuilt-data-dir artifacts/data_condition_minmax/split_seed42/FD004 \
  --output-dir artifacts/temporal_condition_minmax/FD004/stride50_B_seed42 \
  --device cuda --train-stride 50 --validation-stride 50 \
  --epochs 30 --batch-size 128 --learning-rate 0.002 --seed 42 --split-seed 42 --torch-threads 4
OMP_NUM_THREADS=4 PYTHONPATH=src ../.venv/bin/python scripts/audit_condition_normalization.py --dataset FD004
```

数据/训练入口拒绝覆盖已有产物，重跑需独立版本。FD002旧无参数命令默认行为保持。

## 两个多工况子集总结

| 子集 | 全局Test RMSE | 条件化Test RMSE |
|---|---:|---:|
| FD002 | 43.9029 | 18.7242 |
| FD004 | 44.7718 | 20.7756 |

同协议在两个子集都缓解退化。下一步是按新条件化窗口重新预训练MAE、训练对齐、
生成频谱与Qwen缓存，再训练完整TMAF；本次未执行多模态重建，旧缓存不能混用。
