# FD001 窗口48：训练窗口聚类原型条件实验

> 2026-11-03 后续更正：[Observed] GPU与宿主机驱动正常，先前失败来自当前沙箱的私有/dev未暴露/dev/nvidia*。沙箱外PyTorch CUDA运算通过；已启动用户授权的完整实验（PID1820317），seed23/A已完成前两轮，日志`logs/prototype_guidance_full.log`。以下“GPU不可用/未开始”描述保留为初次受限环境检查的历史状态；正式下游成绩仍未产生。首次GPU启动因nvidia-smi进程名称非UTF8在环境记录阶段停止，日志保留；仅修复日志解码后启动，科学配置未改。

[Observed] 实现和 CPU 检查完成，完整 GPU 实验未开始。PyTorch 2.13.0+cu130 的 `cuda.is_available()` 为 false，nvidia-smi 无法连接驱动；两次全量入口均在首个训练前停止（第二次仅补充运行环境审计）。没有70轮权重、正式生成数据或40个正式预测器成绩，不能判断原型能否降低下游 RMSE。

这是**基于训练窗口聚类的原型条件**试验，不是 K-ProtoDiff 完整复现；聚类编号不等同于已知故障类型。范围为用户明确授权的 extension:EXT-PROTO-001，严格论文复现结论保持。排除 FD002 历史追踪、FD001 125组机制、其他数据集、K搜索与额外消融。

## 已核对实现

[Observed] 工作区未找到 AGENTS.md。读取 reproduction-state.md、experiment-ledger.md、历史检查点下游验证记录和最近固定卷积诊断记录；旧生成器见过内部验证发动机，不能用于本次 A 组。

[Code] 原全数据入口为 `upstream/MainCondition.py` → `data/CMAPSSDataset.py` → `TrainCondition.train/sample`。原数据类先拟合全训练文件 MinMax，且初始化会读取原测试文件，因此新入口复用其列定义，独立从 train_FD001.txt 构造训练/验证发动机窗口，避免全数据拟合及不必要的测试读取。每发动机所有连续48周期窗口，窗口末端 RUL=max_cycle−cycle，截断125。14传感器分别 MinMax，新增特征标准化也只用训练窗口。

[Code] 原连续条件直接使用截断后的周期值（不除以125），`classes_emb: Linear(1,32)` → `classes_mlp` → ResNet 条件通路；原 U-Net 保留条件随机丢弃 .2、原整批数值RUL置零 .1、频域模块和原 diffusion pred_noise MSE。数值0是有效编码，与显式空条件不同。本实验 A/B 均保留该历史数值置零规则；B 在该规则触发时仍用窗口自身原型。

[Code] 原 `forward_with_cond_scale` 的条件分支在 eval 时仍默认随机丢弃。本实验两组都使用独立 SamplingConditionMixin，条件分支明确 drop=0，CFG空分支明确 drop=1；后者在共享条件掩码处替换整个 RUL+原型和为原 null_classes_emb，因此空原型贡献为0。原版训练/采样默认文件均未改动。

[Code] 新 `upstream/prototype_guidance.py` 在原 classes_emb 输出上加 learnable Embedding(4,32) → Linear → SiLU → Linear 的投影，然后继续原共享掩码和 MLP。未开启时工厂直接返回原 UNet1D_fre。原型参数在独立 fork_rng 中初始化，避免改变基础参数和全局随机数流。

## 冻结实验协议

- 数据：固定既有30台验证发动机名单，70台训练发动机。训练11056窗口，验证4875窗口。MinMax仅拟合训练发动机原始周期；无裁剪验证值，保留 MinMax 外推行为。未读取原测试文件。
- 原型：每传感器均值、总体标准差(ddof=0)、最后12周期均值−最前12周期均值，合计42维。训练窗口 StandardScaler，KMeans K=4/n_init=10/seed20261103。保存标准化参数、中心、窗口编号、engine_id/end_cycle 和各段频率。五段按原始截断周期0–30/31–60/61–90/91–124/125切分。
- 训练段窗口数分别2170/2100/2096/2113/2577；四原型计数4449/781/1478/4348。实际无空寿命段。空段回退全局训练频率并记录，人工构造空段检查已通过。
- 训练：seed23 的 A/B 然后 seed13 的 A/B；两组每次70轮、batch256、3080更新；预存64和70轮。原 Adam .002、grad_clip1、warmup multiplier2.5/8轮+cosine、DDPM1000/linear保留。共享初始化、训练顺序和扩散随机状态保存哈希并强制核对。cuDNN benchmark=False/deterministic=True；不打开不支持 median 的全局 strict 模式。GPU全训练可重复性尚未实测。
- 生成：每候选11056条，目标 RUL 清单取训练窗口标签，固定相同顺序；batch256/DDPM1000/CFG3/sample seed103，所有候选重置同一扩散随机流。原型用独立 NumPy Generator(seed7103) 从训练分段频率抽样，每条整个去噪过程固定；采样接口不接受参考窗口。RNG哈希用于 A/B 核验，不把原型抽样插入 torch 扩散流。
- 预测器：原 LSTM（14输入/28循环隐藏/单层/48窗口/64全连接隐藏），5种子3/13/23/33/43。相同固定70/30生成数据划分（7739/3317），Adam .002、原15/25/35衰减、batch256，每次固定35轮/1085更新；保留原 epoch>5、每约1/5轮生成dev RMSE选权重规则。为确保预算相同，独立入口关闭原patience提前停止；这是两组对称的协议变化，不能直接对照历史分数。
- 复用现有 `run_fd001_checkpoint_downstream_validation.metrics` 计算总体/分段及逐发动机误差，主指标为先每发动机平均窗口平方误差、再平均发动机和5个预测器、最后开方。输出每预测器配对差值，汇总表并非模型集成预测误差。
- 两个固定轮次均配对报告；若选一个候选，两组分别按同一 pooled engine-balanced 真实验证RMSE选择，平局取较早轮次。该验证既用于比较也用于候选选择，成绩属于探索性结果，不是独立测试收益。绝不使用原测试集。

## 已执行检查

[Observed] `runs/ext-prototype-fd001-w48-20261103/checks.json` 与日志记录：原型关闭精确返回原模型并逐位匹配输出；共享初始参数相同；原型嵌入梯度L1=0.001320，所有投影参数也有非零梯度；联合空分支不受 RUL/原型变化影响；训练/验证发动机无交集；独立原型抽样不改变 torch RNG；两组各100步四样本CPU采样完成，逐前向核对条件/空分支为0/1，无训练随机丢弃；两组各两次配对 Adam 更新，最终 RNG 状态一致；空段回退测试通过。

[Observed] `cpu-downstream-plumbing/` 额外使用64条随机诊断输入跑通原 LSTM、35次更新、生成dev选权重、保存/加载、真实验证预测与既有分段/发动机等权评估函数。它仅检查流程；其中 RMSE 不属于本实验结果，不纳入 summary/paired.csv。实际实验每预测器1085更新未执行。

## 正式结果（尚未产生）

| 生成器种子 | 检查点轮次 | A组RMSE | B组RMSE | B-A | 5个预测器中改善的个数 |
|---|---|---|---|---|---|
| 23 | 64 | 未运行 | 未运行 | — | — |
| 23 | 70 | 未运行 | 未运行 | — | — |
| 13 | 64 | 未运行 | 未运行 | — | — |
| 13 | 70 | 未运行 | 未运行 | — | — |

当前结果：未测量 → 论文无此扩展对照指标 → 数值差距不适用 → 跨生成器种子方向未知 → 不改变严格复现结论。判别分数、条件读出误差也未运行。证据不足，不扩大实验。仅当两个生成器种子的下游方向一致改善，才建议下一轮同RUL段内打乱训练窗口原型编号、保留频率和参数量；本轮不执行该消融。

## 配置、命令与产物

从 `/home/cw_boe/projects/rul` 运行：

```bash
# 已完成；已有快照不应重复 prepare
.venv/bin/python metaindux-ts-reproduction/scripts/run_prototype_guidance.py check
# GPU恢复后执行；自动按seed23再seed13，保存64/70，各5预测器
.venv/bin/python -u metaindux-ts-reproduction/scripts/run_prototype_guidance.py run > metaindux-ts-reproduction/logs/prototype_guidance_full.log 2>&1
# 可重建已完成单元的配对汇总
.venv/bin/python metaindux-ts-reproduction/scripts/run_prototype_guidance.py summarize
```

全新快照的首次准备命令为相同脚本 `prepare`（本工作区已执行，禁止覆盖）。训练若中途产生部分候选后失败，脚本显式停止，先审计，不能把残留权重伪作完整训练或静默重跑。

配置：`configs/extensions/prototype_fd001_w48.json`。
快照：`runs/ext-prototype-fd001-w48-20261103/manifest.json`，含源代码/配置/数据校验和、环境冻结、git差异和状态；`execution_environment.json` 记录运行实际设备、驱动探测及固定卷积设置；`execution_status.json` 明确0次完整训练。
训练输入、验证、原型与条件清单：同一 runs 目录下 `train.npz/validation.npz/normalization.npz/prototypes.npz/plan.npz`。初始检查版本保留 `initial-check/`。
正式结果目录：`results/extensions/ext-prototype-fd001-w48-20261103/`；未来 `seed{23,13}/{A,B}/` 下保存70轮损失/审计、epoch64/70权重、生成样本、各预测器权重/预测/分段与逐发动机误差；顶层 `paired.csv/summary.csv/selected.json`。目前正式结果为空。

停止条件：GPU不可用时不以CPU小实验冒充完整对照；GPU可用后最多4生成器训练/8生成数据单元/40预测器，不扩数据集/K/种子/候选。选择性指标改善或只有一个生成器种子改善均视为证据不足。没有尚待确认的科学配置问题；剩余为恢复GPU后执行已授权命令。
