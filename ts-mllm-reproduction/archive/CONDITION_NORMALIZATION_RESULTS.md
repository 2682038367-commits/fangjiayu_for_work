# FD002：按工况Min-Max单变量验证

后续FD004相同协议已验证完成，时间Test RMSE20.7756（全局44.7718），见
[FD004_CONDITION_NORMALIZATION_RESULTS.md](FD004_CONDITION_NORMALIZATION_RESULTS.md)。下文为FD002阶段记录。

## 结果

GPU训练和独立检查点重放使用NVIDIA GeForce RTX 4060 Ti。两组训练目标均为
RUL/125；以下A/B仅指归一化方式，不是目标尺度对照。

| 归一化 | Val RMSE | Test RMSE | Test MAE | Test Score | Test raw预测std | Test raw预测/标签相关系数 |
|---|---:|---:|---:|---:|---:|---:|
| A：全局每传感器Min-Max | 42.0166 | 43.9029 | 37.8341 | 78193.3769 | 0.7831 | 0.0205 |
| B：每工况每传感器Min-Max | 21.4168 | 18.7242 | 14.1461 | 2379.2979 | 39.8458 | 0.9038 |

按预定验证RMSE选择条件化Min-Max。两组best epoch均30，训练210次参数更新；
模型初态SHA256完全相同：
`b95ee0d0cd7360e8a9046f2280ba03b93eb84bbfa3242b565f1c14dd68d6bc64`。
为补齐旧A未保存初态的证据，本次同配置GPU重跑A，指标精确复现旧A；这是单seed
配对对照，不是3seed稳定性实验，也没有调整轮数或其他训练参数。

主RMSE/MAE/Score按既有协议：预测乘125恢复周期单位，再clip[0,125]计算；
预测裁剪和官方test标签cap仍为复现假设。B raw测试RMSE19.0346、MAE14.7487、
Score2473.5360；clip预测std39.0472。不存在通过单位混用人为降低RMSE的情况。

## 解释与边界

这个配对结果强烈支持：在当前FD002实现与训练预算下，全局归一化是预测退化的
重要因素。B不仅增加预测方差，预测与真实标签的相关系数也从0.0205到0.9038，
不是仅制造随机波动；当前最佳时间模型已摆脱近似常数预测。

不能认为工况归一化是唯一原因，不能将单seed结果称为稳定性证明。论文主表
FD002的14.22属于完整TS-MLLM，当前18.7242是时间基线，不直接视为同模型差距。
论文式(20)明确Min-Max，但未明确按工况拟合，所以本次B是披露的复现假设，
不是已证实的作者原方法；并未把Min-Max换成传感器Z-score。

## 唯一改动及工况识别

使用原A manifest中的208个训练发动机与52个验证发动机，划分不变。
仅在训练发动机的原始三个setting上拟合均值/标准差，再KMeans6（seed42、n_init10）。
设置标准化仅用于识别工况，传感器仍使用每工况Min-Max。验证/测试只分配到最近
训练中心，不重拟合、不使用验证/测试数据更新sensor min/max。

学到的六个中心约为：

| 工况 | setting1 | setting2 | setting3 |
|---|---:|---:|---:|
| 0 | 0.0015 | 0.00049 | 100 |
| 1 | 10.0030 | 0.25049 | 100 |
| 2 | 20.0030 | 0.70051 | 100 |
| 3 | 25.0030 | 0.62050 | 60 |
| 4 | 35.0030 | 0.84050 | 100 |
| 5 | 42.0030 | 0.84048 | 100 |

每个时刻按自己的工况归一化，允许窗口内工况切换；工况ID和三个setting不作为
新增模型输入，仍保持14传感器。常量范围采用安全分母1；不裁剪验证/测试超界
输入。验证/测试超出[0,1]的传感器元素比例约0.0389%/0.0124%。
训练集同工况内传感器std中位数由全局归一化约0.0087扩展到0.1409。
这些识别/分配/常量处理规则也都是补充假设。

其他保持：window40/sample-stride50、Patch4/stride1/dim64/head1、2层Transformer、
平均池化/Linear时间头、Adam/MSE、RULcap125、目标RUL/125、batch128、lr0.002、
30轮、split_seed42/seed42/CPU线程4，按验证RMSE选模。

## 数据与工程核验

train812/val195/test259，所有窗口target/unit/cycle/start_row/left_padding与A逐元素
一致；仅x因归一化变化。独立从原始文件加载、在208个训练发动机上重拟合，
得到的centers/minimum/maximum与保存scaler相同；重新构建全部窗口x和标签一致。
官方259个发动机末窗口与NASA RUL对应、原始文件checksum核验通过。

GPU独立重放A/B验证与测试，指标和CSV输出重现；目标恢复单位、最佳epoch、
源manifest SHA一致。输出scaler.json复制实际导出数据scaler并校验SHA，不再
错误保存训练入口临时构建的全局scaler。旧全局数据、旧模型、Qwen缓存均未覆盖。

44项测试通过，compileall通过。CUDA attention backward仍提示可能非确定性，
本次未运行多seed；3种子稳定性按用户决定暂缓。

## 产物及下一步

- 条件化数据：`artifacts/data_condition_minmax/split_seed42/FD002/`
  含window40/stride50数组、scaler、manifest、normalization_audit。
- B时间：`artifacts/temporal_condition_minmax/FD002/stride50_B_seed42/`
  含best.pt、history/result、selection/output_audit、scaler、测试CSV/SVG。
- 同配置重跑A：`artifacts/normalization_comparison/FD002/global_minmax_seed42/`。
- 可选依赖在pyproject的conditions extra（scikit-learn）声明，当前环境已安装。

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src ../.venv/bin/python scripts/rebuild_condition_data.py
PYTHONPATH=src ../.venv/bin/python scripts/train_temporal.py \
  --dataset FD002 --rebuilt-data-dir artifacts/data_condition_minmax/split_seed42/FD002 \
  --output-dir artifacts/temporal_condition_minmax/FD002/stride50_B_seed42 \
  --device cuda --train-stride 50 --validation-stride 50 \
  --epochs 30 --batch-size 128 --learning-rate 0.002 --seed 42 --split-seed 42 --torch-threads 4
OMP_NUM_THREADS=4 PYTHONPATH=src ../.venv/bin/python scripts/audit_condition_normalization.py
```

本次只验证FD002时间基线，没有修改主数据默认或重跑多模态。
下一步对FD004复制同一归一化协议并先验证时间基线；之后若采用新归一化，
必须重新生成两集MAE/对齐/谱图/Qwen缓存及完整TMAF，不能拼接旧全局归一化缓存
与新条件化时间窗口。数据SHA检查会拒绝混用。
