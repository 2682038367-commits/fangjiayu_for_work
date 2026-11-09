# FD002 seed43最低损失与第70轮权重核查

- [Code] 主对象是刚完成的两次原设置完整训练；较早固定卷积训练只列为库存，未替换本轮对象。用户明确要求若最低损失在第70轮则记录没有形成对照。

| 训练 | 最低损失轮次 | 最低损失 | 与第70轮权重相同 | 形成对照 |
|---|---:|---:|---|---|
| latest_native/repeat1 | 70 | 0.024469953625 | True | False |
| latest_native/repeat2 | 70 | 0.029548320005 | True | False |
| earlier_fixed/repeat1 | 30 | 0.028082059136 | False | True |
| earlier_fixed/repeat2 | 30 | 0.028082059136 | False | True |

- [Observed] 损失历史按原e>5规则与全70轮全局最小值分别复核；best/final70直接逐命名张量比较，不以文件序列化哈希不同误判模型不同。
- [Observed] 本轮主对象两次均为best70=final70，没有可比较的两个模型。生成器重训0次、生成0批、预测器拟合0次；无下游RMSE差值。按用户停止分支结束，不能将其解释成选模有收益或无收益。
- [Observed] 较早固定卷积训练的best30与final70不同，是另一个现成候选对照；本轮未自行切换至该训练，也不新增轮次搜索。
- [Hypothesis] 本轮无法检验最低损失规则是否遗漏更有用的权重；相同权重不会提供该规则与final70的因果对照。
- Snapshot `runs/diagnostic-fd002-seed43-best-final70-inventory-20261106/run-manifest.json`；库存 `results/diagnostics/fd002_seed43_best_final70_inventory_20261106/inventory.json`。当前无对照→无论文对应目标→差距NA→无五种子下游结果→严格复现结论不变。固定核查预算结束。
