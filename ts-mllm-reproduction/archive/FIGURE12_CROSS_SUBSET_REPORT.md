# Fig. 12 prompt管线扩展到FD001–FD004

所有训练与Qwen推理在RTX 4060 Ti完成。原始主模型和缓存未覆盖。FD001保留已修复的stride50 MAE协议；FD002/FD004保留固定工况归一化；FD003使用此前batch32时间基线诊断版本，其余子集沿用主方案batch128。对每个子集，旧文本A与新Fig.12版本的数据manifest、时间权重、TMAF参数初始哈希、网络结构及训练设置相同；仅prompt管线更换（需要重新对齐、生成频谱融合权重并重建Qwen缓存）。并非单纯替换prompt字符串的严格因果消融。全部训练30轮，按验证RMSE保存模型。

新文本包含Fig.12示意的任务描述、数据集描述、min/max/median、整体趋势及结合视觉特征分析指令；统计聚合、趋势判定与精确措辞仍为复现假设。FD002/FD004的实际三维运行设置没有逐周期写入这个Fig.12 profile，不能把它称为完整工况知识。512-token预算无截断，缓存标签/发动机/末周期映射通过。

| 子集 | 归一化 | batch | 旧A验证RMSE | Fig.12验证RMSE | 旧A测试RMSE | Fig.12测试RMSE | 旧A Score | Fig.12 Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 全局Min-Max | 128 | **17.06** | 17.18 | **16.32** | 17.82 | **575** | 776 |
| FD002 | 工况Min-Max | 128 | **18.30** | 19.39 | 17.09 | 17.07 | **1551** | 1801 |
| FD003 | 全局Min-Max | 32 | 17.74 | **17.49** | 22.08 | **19.25** | 2506 | **1627** |
| FD004 | 工况Min-Max | 128 | **17.58** | 18.19 | **19.00** | 20.17 | **2465** | 5424 |

判断只依据验证集：Fig.12管线仅FD003有RMSE改善，FD001、FD002、FD004的验证RMSE均较差；不替换固定主方案。测试数字仅是已选模型的描述性结果，不能用于选择prompt。FD002测试RMSE差0.02周期不足以抵消验证集恶化。单种子不代表稳定性结论。FD003仍显著高于论文完整模型11.97。

完整哈希链、训练配置、对齐与测试对应核验见`FIGURE12_CROSS_SUBSET_AUDIT.json`；复核脚本`scripts/audit_figure12_cross_subset.py`。新产物位于`artifacts/figure12_v1/FD001|FD002|FD004/seed42/{alignment,qwen,spectrum,tmaf}`，FD003位于`artifacts/fd003_figure12_v1/seed42/{alignment,qwen,spectrum,tmaf_batch32}`。FD002首次训练在交互切换时中断，部分checkpoint保留于`artifacts/figure12_v1/FD002/seed42/tmaf_interrupted_20260920`；完整重跑的`tmaf`目录具有30轮history/result。

## 逐token塌缩来源的只读核验

FD003新缓存92个验证窗口：每个有效token数165–166，padding为0，mask与窗口对齐；重建的训练前模型初始SHA256与记录一致。训练前3496次时间patch查询的平均最大token权重仅0.0286、注意力熵均值4.878、Top-1/次高logit间距中位数0.077；最高token位置分布于多个位置。保存的最佳模型中全部3496次查询都选索引3（固定标题`Describe`），平均最大权重0.999999、熵约1.15e-5、logit间距中位数19.68。缓存token3的范数中位数102.69，低于其他有效token范数中位数110.51；训练后Key3范数53.31，高于其他Key范数中位数28.92。故塌缩是**训练后形成的投影/logit饱和现象**，不是缓存一开始就只含一个强token，也未发现mask错位。范数与logit统计不唯一确定因果根源；不擅自修改温度或投影并声称符合论文。

只读详细结果`FD003_TOKEN_COLLAPSE_DIAGNOSIS.json`，脚本`scripts/diagnose_fd003_token_collapse.py`。逐token版本是非论文公式的补充诊断，验证收益极小、测试RMSE更差，现阶段停止调优。全局广播的均匀注意力是论文公式字面上的结构性问题，不会通过prompt本身消除。
