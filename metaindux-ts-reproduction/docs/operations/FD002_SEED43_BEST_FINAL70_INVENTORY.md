# FD002 seed43：核查刚完成训练的best与final70

Scope `scope-20261106-fd002-seed43-best-final70-inventory`，diagnostic/supporting。用户授权复用刚完成训练中同一次训练的最低损失权重与第70轮权重，若相同则记为没有形成对照。

- [Observed] 最近完成的stage-attribution实验保存了两次原生seed43完整训练。两次训练均同时保留best、67、70。库存另列较早固定卷积设置两次训练，不能因其best轮次不同而悄悄替换对象。
- [Code] 直接读取每次70轮训练历史，分别检查全局最小值与原规则e>5的最小值；复核audit中的best轮次、损失、文件哈希。CPU逐命名张量比较best与final70，避免序列化容器差异被误认为权重差异。
- [Code] 若本轮主对象best恰为70且权重相同，按用户明确分支结束：0生成器重训、0新生成、0预测器拟合，无RMSE对照。记录所有候选身份，不跨训练混用best与final。
- [Hypothesis] 相同权重没有提供选模规则的对照，既不能支持“第70轮更好”也不能支持“第70轮更差”。其他旧训练能否形成对照是另一个对象选择，不因查到可用库存自动切换。

严格main/A/B结论保持、C推迟；不访问官方测试集，不新增轮次或种子。此CPU只读核查无实质训练成本，预算一次库存核查。执行`.venv/bin/python analysis/diagnostics/check_fd002_seed43_best_final70_inventory.py`。
