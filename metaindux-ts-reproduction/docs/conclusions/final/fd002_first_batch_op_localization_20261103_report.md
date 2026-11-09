# FD002首批最早不同算子定位（2026-11-03）

[Observed] **已定位到CUDA卷积的数值执行：相同输入、权重、偏置、形状和布局，卷积输出首次不同。最早出现在哪一层会随重放进程变化，不能称为唯一固定异常层。** 本轮3次首批前向、4次独立卷积调用，0反向、0参数更新；没有完整训练、生成或下游评价。

|对照|最早不同模块|ATen算子|之前已跟踪输出逐位相同的数量|
|---|---|---|---|
|公开入口 vs 审计入口|`minusTrend.fn.seasonal.season`|`aten.convolution.default`|721|
|补充捕获 vs 前述任一入口|`mid_block1.block1.proj`|`aten.convolution.default`|310|

## 主对照：季节性分支1×1卷积

[Observed] 全部779个已跟踪操作中，第722个操作首次不同。模块`minusTrend.fn.seasonal.season`是网络末尾季节性/周期分支的1×1卷积：输入`[256,32,48]`，权重`[16,32,1]`，偏置`[16]`，输出`[256,16,48]`。输入值、权重、偏置、stride/padding/dilation/groups、内存步长与偏移全部一致。算子执行之前未发现已跟踪的值差异。

[Code] 定义：`upstream/DiffusionFreeGuidence/Attetnion.py:187`；调用：`:197`。这是实际执行卷积的首个差异点，而不是把某个接收到上游偏差信号的后续层误判为起点。具体CUDA/cuDNN内核名称尚未测量，不冒充已经识别。

## 补充捕获：分叉可以更早出现

[Observed] 为保存主对照算子输入/输出，额外进行一次首批捕获。初始全状态、第一批数据/随机题目仍与冻结记录一致，但这次最早在第311个操作、`mid_block1.block1.proj`卷积分叉，输入`[256,64,24]`、权重`[64,64,3]`，对应`upstream/DiffusionFreeGuidence/Unet1D_fre.py:136`的`F.conv1d`。这个卷积的输入和标准化后权重也逐位一致。

[Observed] 因为更早的卷积已分叉，这次后续第722操作的输入不再等于主对照中的输入。“原样回放”逐元素保护检查正确停止，原失败日志/快照保留。未放宽相等要求，也未继续挑选重放进程。两个中位数操作位于已跟踪索引498/663；主对照它们输出相同，补充对照在它们之前已分叉。因此前次median不支持严格模式的报错不能当作本次首个分叉算子。

## 局部卷积检查（独立标注）

[Observed] 使用补充捕获的季节性卷积输入，另行冻结为新的局部输入，两独立进程各执行一次普通模式与一次确定性模式，未再进行整网前向。它不是主对照第722操作的精确激活回放。

|局部模式|两进程结果|
|---|---|
|benchmark=True / deterministic=False|逐位相同|
|benchmark=False / deterministic=True / strict deterministic algorithms|逐位相同|
|两种模式之间|最大绝对差4.76837158203e-07；不同元素123844/196608|

[Observed] 这证实卷积计算模式可以改变浮点舍入结果；但本次局部普通模式也重复一致，**不能宣称关闭自动选算法已消除原整网分叉**。整网严格确定性仍受前次median算子限制；本轮未替换算子或修改训练默认值。

## 结论与停止点

[Observed] 当前相同随机题目下的前向分叉至少可发生于两个卷积实例。将后续核查重点收缩到卷积算法/计算精度选择，比继续把差异归因于样本顺序或条件抽样更有直接证据。

[Hypothesis] cuDNN自动选算法、执行上下文或计算精度路径差异是候选解释，尚未直接记录所选内核，所以不唯一归因于benchmark，也不能推断原第67轮DS差距全由它造成。追踪同步本身会改变时序。

[Hypothesis] 若继续，应单独检验卷积算法选择规则对首批前向重复性的影响，并记录实际内核；无须重训70轮。该后续未启动，不自动改变模型/训练/选模规则。

[Observed] 本诊断→无论文同设计指标→论文数值差距不适用→当前入口有限配对，不是生成种子方差统计→严格复现结论保持。FD003/4及125组分析仍暂停，seed43下游门槛与seed13结果保持。

快照：`runs/diagnostic-fd002-first-batch-op-localization-20261103/run-manifest.json`；失败保护检查：`runs/diagnostic-fd002-first-batch-op-localization-20261103/confirmation-manifest.json`；独立局部检查：`runs/diagnostic-fd002-first-batch-op-localization-20261103/local-probe-manifest.json`；原始算子记录/核验：`results/diagnostics/fd002_first_batch_op_localization_20261103`。
