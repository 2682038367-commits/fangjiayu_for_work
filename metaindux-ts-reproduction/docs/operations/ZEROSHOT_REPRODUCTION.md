# Table V 零样本复现

2026-10-14：用户要求在已有少样本复现基础上增加零样本复现，恢复 Candidate B。

## 范围与来源

- **[Paper]** Section IV-D（印刷页 18071）：生成器只使用 RUL > 30 的正常样本；生成数据训练两层 LSTM，仅在真实疲劳样本上测试。目标为 Table V（PDF 第 9 页、印刷页 18070）的 MetaIndux-TS 列：FD001 6.353、FD002 22.574、FD003 11.289、FD004 35.062，报告平均 18.820。
- **[Code]** 沿用 `SOURCE_SNAPSHOT.md` 固定的作者源码及当前少样本路线：w48、14 传感器、train-fit MinMax、RUL cap 125、DiffUnet_fre、hard_random_quantile、T=1000、linear DDPM、Adam lr=0.002、warmup_cosine、70 epochs、batch 64。除训练样本选择从“正常 + 10% 疲劳”变成“仅正常”外，保持配置一致。
- **[Code]** 原始 `upstream/main_fewshot.py` 已实现 zeroshot；`upstream/script/train_zero_diff.sh` 虽名为 zero，却配置了 fewshot，不能直接用作零样本入口。
- **[Code]** RUL=30 按现有协议归入疲劳（测试条件 <=30）；论文只写 >30 / <30，未定义等号。归一化仍按作者代码使用全部训练周期拟合，采样仍使用完整训练标签；生成器梯度更新只接触正常窗口。零样本在这里指疲劳特征未用于生成器训练，并非完全不使用疲劳标签或统计量。
- **[Code]** 5 个生成种子和 5 个评估种子均为 3/13/23/33/43，每数据集 25 次正式评估，synthetic 70/30 固定划分 seed=20260915。种子 ID 和固定划分为本地补全。额外 DS 为沿用评估器产生的 supporting diagnostic，不属于 Table V 的目标指标。

范围修订 `scope-20261014-zeroshot`：strict-reproduction，core:B，依赖关系为 shared-base but outcome-independent。不重跑 Table I / VI，不增加消融或调参；只覆盖 MetaIndux-TS 一列，不能据此验证相对其他模型的排名或机制。若配置或数据校验失败则停止；不因指标不匹配而自动调整参数。

## 执行

在可访问 CUDA 的终端执行完整 20 次生成训练及 100 次正式评估：

```bash
bash scripts/run_paper_zeroshot_full.sh
```

先查看生成命令：

```bash
.venv/bin/python scripts/run_experiments.py --config configs/paper_zeroshot_full.yaml --dry-run
```

先完成 FD001 的正式矩阵：

```bash
.venv/bin/python scripts/run_experiments.py --config configs/paper_zeroshot_fd001.yaml --skip-completed
TF_ENABLE_ONEDNN_OPTS=0 TF_DETERMINISTIC_OPS=1 .venv/bin/python scripts/evaluate_fd002_stability.py --config configs/evaluation_fd001_zeroshot.yaml
```

训练耗时未测量；计算量为 20 个 70-epoch 生成器及其 1000-step 采样，另加 100 个 35-epoch LSTM 和评估器内置判别器训练。CUDA 不可用时完整启动脚本在训练前报错。

## 产物和判定

- 训练权重、合成数据、loss 和 metadata 使用 `zeroshot_paper` 后缀，与既有实验隔离。
- `results/zeroshot/splits/*_indices.npz` 记录正常/疲劳索引，`selected_fatigue_indices` 必须为空。
- `results/runs/*_zeroshot_paper_metadata.json` 记录实际命令、完整配置、环境、时间和执行状态。
- `results/zeroshot/*_table_v_evaluation_{raw,summary}.csv`、split 和 manifest 保存正式评估；raw CSV 支持逐项恢复。
- 完整启动脚本最后调用 `analysis/audit/summarize_zeroshot.py`，写 `results/zeroshot/table_v_comparison.csv`。缺少数据集或出现重复/不完整评估时拒绝汇总。
- 将差距与同协议多生成种子变化比较，逐数据集报告“本地 → 论文 → 差距 → 是否超出种子变化 → 是否改变结论”；不能仅凭接近论文数字宣布机制复现。

**[Observed]** 当前仅完成代码接入和数据划分诊断，尚无零样本训练或 RMSE 结果。准备快照为 `runs/core-b-zeroshot-preparation-20261014/run-manifest.yaml`；正式运行结果须另补实际执行快照及 ledger 后再纳入证据。已有冻结报告保持其历史含义。
