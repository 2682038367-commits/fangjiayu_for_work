# 优先级4/5：FD001完整文本缓存、频谱重建与新协议训练

已使用新window40/sample-stride50数据和DKE96/线性projector完成FD001闭环，全部模型
前向和训练使用RTX4060Ti。旧stride1频谱、Qwen缓存与模型未覆盖，未用旧时间分支权重
作为新TMAF初始化。FD002–004最终多模态缓存和训练尚未完成。

## 优先级4：512文本及完整输出

- max_text_tokens=512，通过真实DKE96→线性桥接→Qwen，不直接绕过DKE；
- 保留全部有效文本输出及1个视觉前缀，缓存容量513，不再截取末31个；
- 当前prompt实际文本长度237–244，因此含视觉的有效token为238–245；512是上限，
  不是强制生成512个有意义token，剩余状态补零且mask=False；
- 缓存float16，Qwen BF16推理；这种精度和冻结策略仍为工程假设。

| Split | Qwen缓存 | 频谱缓存 |
|---|---|---|
| train | [309,513,1024] | [309,3,114,114] |
| val | [76,513,1024] | [76,3,114,114] |
| test | [100,513,1024] | [100,3,114,114] |

## 优先级5：重建与来源核验

从新窗口在线重新生成RP/STFT/CWT，使用新对齐阶段学到的sensor fusion；持久化为
float32、scale=1，不使用旧uint8量化谱图。读取器支持manifest scale=1和旧scale=255，
防止新浮点图错误地再次除以255。

新Qwen缓存约487MiB、新频谱约73MiB，分别位于：

- artifacts/qwen_audited/FD001/seed42/
- artifacts/spectrum_audited/FD001/stride50_seed42/

保存mask、target、unit、cycle；SHA256记录原始窗口manifest、对齐checkpoint、MAE
checkpoint、每个缓存文件，Qwen快照commit=c1899de289a04d12100db370d81485cdf75e47ca。
完整性核验通过：有限值、dtype/shape、零padding、每行完整prompt长度、发动机/周期/
标签映射、频谱来源一致、全部文件checksum。新TMAF加载时再次检查窗口与Qwen及
时间checkpoint来源，避免新旧协议混用。

## 新协议训练

先从头训练新stride50时间分支，再加载其最优checkpoint训练字面global_broadcast
TMAF。两阶段各30epochs、batch128、Adam/MSE、统一lr0.002；TMAF不冻结热身，
temporal lr也是0.002。Patch4/stride1/dim64/head1、attention key64/output32、MLP512/
dropout0.5保持论文明确数值。按验证RMSE选checkpoint（此选择协议是未公开假设）。

| FD001 seed42 | 最优epoch | Val RMSE | Test RMSE | Test MAE | Test Score | 预测std |
|---|---:|---:|---:|---:|---:|---:|
| 新时间基线 | 30 | 74.9039 | 66.2019 | 55.4415 | 63813.18 | 0.0031 |
| 新字面TMAF | 27 | 24.0574 | 24.0989 | 18.8640 | 4014.93 | 33.3758 |
| 论文TS-MLLM | — | — | 12.45 | 主表未给 | 233.40 | — |

模型产物独立保存在temporal_audited/FD001/stride50_seed42与
tmaf_audited/FD001/global_broadcast_stride50_seed42，包含best.pt/history/result/预测CSV/SVG。
每组均有100条有限测试预测，但时间基线**几乎退化为常数，模型质量验收失败**。
TMAF输出非恒定，但距离论文指标很远，因此不能称性能复现成功。

## 当前问题与后续边界

1. train309/batch128每轮仅3次更新，30轮共90次；旧stride1配置每轮105次，共3150次。
   时间基线训练误差仍约75，表明当前未公开的时间头/初始化/标签尺度等选择没有
   在90次更新预算下学好。不能据此擅自改stride、batch、lr或epoch。
2. 论文的同一global向量复制成Key会使注意力均匀；masked mean全局聚合是披露假设。
   不能仅为效果好而换成token注意力并称其为论文明文。
3. DKE桥接/对齐目标、prompt模板、MAE来源、Qwen型号/冻结、回归头/优化器/位置
   初始化、具体TFT参数仍未唯一确定。低embedding对齐MSE不证明有效RUL推理。
4. 数据验证stride50、官方test标签cap、endpoint/短序列协议也仍需披露。

下一步可以在不改变明确设定的前提下，排查时间分支未公开部分（回归头、初始化、
训练目标数值尺度），只用验证集做选择，记录新版本并重新比较三个seed。
不能直接引用之前12.5539或13.37作为此新协议结果。

## 运行入口

```bash
PYTHONPATH=src ../.venv/bin/python scripts/run_audited_fd001.py
```

入口先缓存、再时间基线、再TMAF；拒绝覆盖已完成产物，不会自动静默重跑。
本次模块回归测试30项通过，训练与缓存验收另按上表如实区分。
