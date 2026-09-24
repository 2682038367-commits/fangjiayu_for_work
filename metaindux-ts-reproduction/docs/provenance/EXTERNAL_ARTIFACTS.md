# 未随 Git 仓库发布的大型产物

为避免将数据集、模型权重和合成样本提交到 GitHub，本仓库刻意不包含以下本地产物：

| 路径 | 原因 | 用途 |
|---|---|---|
| `data/` | C-MAPSS 原始/处理数据的分发与体积限制 | 训练、评估 |
| `outputs/` | 合成样本约 4.5 GB | `audit_checklist.sh` 的 A6 存在性检查 |
| `checkpoints/` | 训练 checkpoint 约 201 MB | 权重复用、θ checkpoint 核查 |
| `upstream/weights/` | 作者/本地产生的权重与合成数据 | 采样 |
| `logs/`、`wandb/`、`.venv/`、`.cache/` | 机器本地运行状态 | 不属于可发布源码 |

轻量、可核查的结果（正式 raw/summary CSV、split、manifest、冻结数字、配置、代码和
文档）已纳入版本控制。因此，未恢复 `outputs/` 时运行：

```bash
bash analysis/audit/audit_checklist.sh .
```

会只在 A6 报告合成 `.npz` 缺失；这不表示 CSV 汇总、统计或代码链路出错。恢复产物后，
审计预期为 `PASS 63 / FAIL 0 / INFO 8`。
