# MetaIndux-TS 复现：最终核对总纲

> **统计口径更新（2026-09-24）**：论文报告的是用生成数据训练的预测分数，未报告
> 论文自己的真实数据训练基线。因此“协议层”、论文生成数据相对论文真实数据的惩罚、
> 生成端占比、跨论文 p 值、Q/I² 及跨数据集协议差异都不可识别，旧版相关结论已撤回。
> 权威机器可读来源：`results/frozen_numbers.json` 的 `observable_comparisons`。

## 最终结论

1. FD002 上，固定 θ=0.25 和公开代码随机 θ 的本地七生成种子配对比较未检出稳定优势：
   RMSE 差（固定 − 公开）为 −2.3181，配对 t 检验 p=0.184，Wilcoxon p=0.219。
2. 本地生成数据相对本地真实数据的 RMSE 惩罚为 +1.597 至 +5.370；这是同一预测器、
   划分与测试集内的可比生成质量诊断。
3. 正式复现未达到论文报告的生成数据训练 RMSE 点估计；差距为 +1.409 至 +6.700。
   这不能再被分解为“协议”或“论文生成质量”的部分。

## 可观测主结果（窗口 48）

| 数据集 | 论文生成数据 RMSE 点估计 | 本地生成数据 RMSE | 本地真实数据 RMSE | 正式复现差距 | 本地生成数据惩罚 |
|---|---:|---:|---:|---:|---:|
| FD001 | 13.949 | 15.358 | 13.760 | +1.409 | +1.597 |
| FD002 | 24.565 | 28.295 | 23.384 | +3.730 | +4.911 |
| FD003 | 16.206 | 20.717 | 15.691 | +4.511 | +5.027 |
| FD004 | 26.577 | 33.277 | 27.906 | +6.700 | +5.370 |

- 正式复现差距 = 本地生成数据训练 RMSE − 论文生成数据训练 RMSE。
- 本地生成数据惩罚 = 本地生成数据训练 RMSE − 本地真实数据训练 RMSE。
- 论文真实数据训练基线、论文生成数据惩罚和论文/本地协议差异均为未知量。

## FD002 阈值消融（本地结果）

| 对比 | 生成种子 | RMSE 差（固定 θ=0.25 − 公开随机 θ） | 结论 |
|---|---:|---:|---|
| 配对评估 | 7 | −2.3181；t p=0.184；Wilcoxon p=0.219 | 未检出稳定优势 |

该消融仅比较本地实现的两个臂，不是对论文阈值机制的因果验证。

## 权威输入与复算入口

| 路径 | 用途 |
|---|---|
| `results/fd001_w48_public_code_evaluation_summary.csv` | FD001 本地生成数据结果（5 个生成种子 × 5 个评估器种子） |
| `results/frequency_threshold_validation/fd002_public_arm_seven_seeds_evaluation_summary.csv` | FD002 公开代码臂的逐生成种子均值 |
| `results/fd003_w48_public_code_evaluation_summary.csv` | FD003 本地生成数据结果 |
| `results/fd004_w48_public_code_evaluation_summary.csv` | FD004 本地生成数据结果 |
| `results/real_data_baseline_w48.csv` | 本地真实数据基线（每数据集 5 个评估器种子） |
| `analysis/audit/gen_frozen_numbers.py` | 生成/校验冻结 JSON |
| `analysis/diagnostics/protocol_gap_decomposition.py` | 输出可识别的比较表 |
| `analysis/audit/audit_checklist.sh` | 一致性检查 |

运行：

```bash
.venv/bin/python analysis/audit/gen_frozen_numbers.py --check
.venv/bin/python analysis/diagnostics/protocol_gap_decomposition.py
bash analysis/audit/audit_checklist.sh .
```

## 报告限制

- 论文没有给出真实数据训练的基线，不能估计本地与论文之间的真实协议差异。
- 论文只提供点估计时，不能把它当成有本地种子方差的随机样本做 t 检验或异质性检验。
- 本地真实基线是诊断参照，不能替代论文缺失的真实基线。
- 任何新训练、重新采样或评估后，先更新原始 CSV，再运行冻结数字生成器和审计脚本。
