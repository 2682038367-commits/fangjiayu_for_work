# FD002 seed43训练、采样、判别三阶段验证

- [Observed] 两次新增原生70轮训练是否完全一致：False；首次损失不同的更新：1；首次参数不同的轮次：1。此前固定设置两次完整训练一致。
- [Observed] 相同旧权重、相同实际噪声的两次固定路径完整采样：{'data_bit_identical': True, 'labels_bit_identical': True, 'different_elements': 0, 'max_abs': 0.0}。
- [Observed] 重复判别评价：{'exact_equal': True, 'max_abs_difference': 0.0, 'interpretation': 'Exact same-data evaluator repeat only if the sample array equality gate passes.'}。
- [Observed] historical_a: meanDS=0.435170920, five=[0.4670197400096293, 0.4327154549831488, 0.4186928261916225, 0.4214010592200289, 0.4360255175734232]。
- [Observed] historical_a_repeat: meanDS=0.435170920, five=[0.4670197400096293, 0.4327154549831488, 0.4186928261916225, 0.4214010592200289, 0.4360255175734232]。
- [Observed] native_rerun_b: meanDS=0.177166586, five=[0.1656836783822821, 0.1625541646605681, 0.2403105440539239, 0.1712806933076552, 0.1460038517091959]。
- [Observed] 共同采样与评价设置下旧权重效应：{'mean_B_minus_A': -0.25800433317284555, 'paired_direction_count': 5, 'descriptive_screen_passed': True, 'caveat': 'Five paired discriminator seeds, one generated batch per checkpoint; descriptive screen, not independent generator replications or full historical causal recovery.'}。
- [Hypothesis] 若采样/评价重复通过且旧权重分数仍不同，支持权重本身承载差距；结合先前卷积内核实验及本次完整训练对照可收缩原因。缺失历史逐步状态，不能重建历史每一次随机/计算选择，也不能声称唯一解释全部历史分数。
- [Code] 每个阶段的对照只改一个因素；无下游预测器训练、其他种子或默认值修改。原生重复未分叉时报告零结果，不继续抽进程寻找分叉。
- Run `runs/diagnostic-fd002-seed43-variation-stages-20261105/run-manifest.json`，结果 `results/diagnostics/fd002_seed43_variation_stages_20261105/summary.json`。诊断→无论文对应目标→差距NA→两次本地训练/单生成批次限制→严格复现结论不变。固定预算结束。

## 2026-11-06完成后独立复核

- [Observed] 75项输入、63项输出、6个检查点哈希全部通过。重读全部更新损失、逐轮状态和原始命名权重，确认原生设置两次初始参数/训练顺序/置零抽签/逐轮RNG状态一致，但第一批损失和后续权重不同。固定设置两次一致的既有证据保持。
- [Observed] 直接对比两个完整A生成数组，0元素不同；从原CSV重算五个重复评价分数完全相同。两个旧检查点均核验实际噪声anchor和采样flags，A/B五次分数均保留差距。
- [Observed] 旧原生采样A/B均值.434123736/.178009148；共同固定采样A/B均值.435170920/.177166586。差距从.256114588变为.258004333，量级几乎不变。
- [Hypothesis] 这组对照支持主要差距由训练得到的不同权重承载；结合原生/固定完整训练与既有实际卷积内核干预，支持卷积路径选择导致训练不可重复。当前证据不支持采样/判别波动解释该大幅差距。仅限当前旧权重对、固定噪声和评价配置，不声称恢复缺失的历史逐步状态或已证明下游收益。
- 复核记录：`runs/diagnostic-fd002-seed43-variation-stages-20261105/completion-verification.json`。预算已关闭，无新增运行或默认设置修改。
