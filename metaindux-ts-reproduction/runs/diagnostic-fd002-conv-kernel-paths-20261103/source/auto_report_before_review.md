# FD002实际卷积内核与固定路径对照（2026-11-03）

[Observed] 本轮直接用Kineto/CUDA活动记录实际卷积内核。每个入口先完成一次相同题目的算法选择前向，再恢复完整随机状态和题目记录，测量缓存路径下前向；四进程共8次首批前向，0反向/0参数更新。预检另执行一次固定输入卷积以验证CUDA活动采集。

|卷积设置|首个输出不同位置|两入口全部卷积内核序列|全部已跟踪输出|
|---|---|---|---|
|自动选择|init_conv|存在不同|存在不同|
|固定卷积设置|全779个操作一致|一致|一致|

## 自动选择模式的直接证据

[Observed] 首个不同操作：aten.convolution.default，模块`init_conv`，索引21。输入、权重、偏置及布局相同；最大输出差4.76837158203125e-07，不同元素217433/393216。实际内核序列是否相同：False。

公开入口实际内核：

- `void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)`
- `void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)`
- `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x32x16_stage1_warpsize4x1x1_g1_tensor16x8x8_aligna4_alignc8_execute_kernel_cudnn_infer`
- `void cudnn::ops::nhwcToNchwKernel<float, float, float, true, false, (cudnnKernelDataType_t)0>(cudnn::ops::nhwc2nchw_params_t<float>, float const*, float*)`
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})`

审计入口实际内核：

- `void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)`
- `void cudnn::ops::nchwToNhwcKernel<float, float, float, false, true, (cudnnKernelDataType_t)2>(cudnn::ops::nchw2nhwc_params_t<float>, float const*, float*)`
- `sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage3_warpsize2x2x1_g1_tensor16x8x8_execute_kernel_cudnn_infer`
- `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})`

[Observed] 本次确认了计算路径贡献：自动模式在同一卷积输入下使用不同实际内核并产生不同输出；固定卷积设置后，两入口实际卷积内核序列和全部779个跟踪输出逐位一致。[Hypothesis] 支持卷积自动选路径/浮点舍入是当前首批分叉的原因；仍不能唯一解释历史第67轮生成质量差距。

[Code] 唯一受控因素为卷积选择设置：native benchmark=True/deterministic=False；fixed benchmark=False/deterministic=True。TF32、数据、初始权重、随机题目、模型算式和环境保持；全局torch严格确定性在所有组均为False，median原实现保留。固定设置是否得到相同路径以测得内核为准。

[Observed] 只把record_function卷积标记的CPU后代所附GPU实际kernel活动计入该次卷积；算法搜索在未计时warmup中完成，避免把搜索候选误当成最终实际执行路径。原始Chrome traces、cuDNN日志、每个卷积的完整输入/输出和内核名保存。

[Hypothesis] Profiler/同步和warmup可改变算法选择，当前是受控重放而非恢复历史内核。即使本轮首批分叉消失，也不等于整条训练/生成严格可重复，更未验证历史DS或下游收益得到修复。

[Observed] 本轮结束，不修改上游默认代码、不训练70轮或生成/下游评估。严格main/A/B结论保持、C推迟，FD003/4与125分析暂停。诊断→无论文同设计指标→论文数值差距不适用→有限入口配对→严格复现结论不变。

快照：`runs/diagnostic-fd002-conv-kernel-paths-20261103/run-manifest.json`；完整结果/原始profile：`results/diagnostics/fd002_conv_kernel_paths_20261103`。
