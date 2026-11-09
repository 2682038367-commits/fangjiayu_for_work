# FD002 seed43固定卷积路径完整训练验证

- [Observed] 完整70轮重复检查通过：True。每次11410次更新；逐更新损失、逐轮完整参数/优化器状态/RNG、训练顺序、学习率及best/67/70权重对比见结果。
- [Code] 只把训练卷积设置改为benchmark=False、deterministic=True；全局strict=False，保留原median、数据、网络、阈值、条件丢弃、优化器及70轮预算。采样设置与已有对照相同。
- [Observed] historic_epoch67: DS=0.434123736；五次=[0.4654549831487723, 0.4288035628310063, 0.4209797785267212, 0.4211603273952816, 0.4342200288878189]。来源 `results/diagnostics/fd002_final70_two_seeds_20261102/seed43/original/evaluation_checkpoint43_sample103.csv`。
- [Observed] native_rerun_epoch67: DS=0.178009148；五次=[0.1654429465575349, 0.162253249879634, 0.250601829561868, 0.1686928261916225, 0.1430548868560424]。来源 `results/diagnostics/fd002_final70_two_seeds_20261102/seed43/phase/evaluation_checkpoint43_sample103.csv`。
- [Observed] fixed_repeat1_epoch67: DS=0.241706789；五次=[0.2712445835339432, 0.1690539239287434, 0.1930067404910929, 0.3435844968704863, 0.2316441983630236]。来源 `results/diagnostics/fd002_fixed_conv_full70_20261105/repeat1/seed43/phase/evaluation_checkpoint43_sample103.csv`。
- [Hypothesis] 完整训练可重复若成立，支持固定卷积路径消除了本环境这两次训练的已测分叉；不能唯一解释历史判别差距，也不能保证分数必然改善。原始历史训练中间状态未保存；沿用的native采样仍可能存在数值路径差异。没有新增下游验证。
- 诊断结果→无论文对应目标→差距NA→仅两次本环境重复→严格复现结论不变。预算到此停止，不自动改默认配置、扩种子或运行下游。
- Run `runs/diagnostic-fd002-fixed-conv-full70-20261105/run-manifest.json`；结果 `results/diagnostics/fd002_fixed_conv_full70_20261105/summary.json`。

- [Observed] 完成后独立复核64项输入、43项输出、6个检查点哈希均通过；重新逐张量比较best30/67/70、核查全部11410个损失及70轮状态记录，汇总分数直接从原CSV重算一致。核验记录 `runs/diagnostic-fd002-fixed-conv-full70-20261105/completion-verification.json`。
