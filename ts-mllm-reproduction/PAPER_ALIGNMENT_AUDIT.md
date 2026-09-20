# 论文—实现逐项核对

当前主流程已恢复式(20)所述全局逐传感器 Min-Max，不使用按工况归一化；
此前将条件化 Min-Max 描述为仅拟合范围的假设不够准确，它还改变了归一化分组。
旧条件化结果保留为额外实验。新工况知识 prompt 的资料来源、窗口摘要及假设边界见
[GLOBAL_KNOWLEDGE_PROMPT.md](GLOBAL_KNOWLEDGE_PROMPT.md)，尚未训练新版本。

最新：FD002/FD004按条件化Min-Max已重建MAE/对齐/谱图/Qwen并完成来源核验，
见[CONDITION_MULTIMODAL_REBUILD.md](CONDITION_MULTIMODAL_REBUILD.md)。未训练新TMAF，
下文“多模态尚未重建”是该阶段完成前的历史状态。

FD004同一工况归一化协议已验证：时间Test RMSE20.7756，见
[FD004_CONDITION_NORMALIZATION_RESULTS.md](FD004_CONDITION_NORMALIZATION_RESULTS.md)。
FD002/FD004均支持条件化Min-Max，但仍是未明确拟合范围的假设，多模态尚未按此重建。

最新归一化专项：FD002同初值单变量对照支持每工况Min-Max，时间Test RMSE18.7242，
详见[CONDITION_NORMALIZATION_RESULTS.md](CONDITION_NORMALIZATION_RESULTS.md)。仍为论文
未明确的拟合范围假设；完整多模态尚未按此重建，旧退化描述对应全局归一化历史结果。

最新：FD002–FD004已完成独立MAE/对齐/新Qwen缓存及B时间/完整TMAF，
四集覆盖、指标与剩余协议差异见[FOUR_SUBSET_MULTIMODAL_RESULTS.md](FOUR_SUBSET_MULTIMODAL_RESULTS.md)。
用户决定暂缓稳定性并停止bias调优；默认bias、B尺度固定。下文未覆盖四集描述为历史状态。

最新回归头/初始化专项核验见[HEAD_INITIALIZATION_AUDIT.md](HEAD_INITIALIZATION_AUDIT.md)：
表II的512与dropout0.5一致，完整层次/激活/pooling仍未唯一确定，结构不改。
两层Transformer构造时初值相同但无参数共享，B训练后已分化；初始化仍属假设。
当前B完整TMAF结果见[TMAF_SCALE_B_RESULTS.md](TMAF_SCALE_B_RESULTS.md)，下文指标为历史状态。

最新更新：优先级4/5已在FD001生成新完整Qwen/频谱缓存并训练新时间分支及字面TMAF，
详见[AUDITED_FD001_RESULTS.md](AUDITED_FD001_RESULTS.md)。下文未重建/训练是初次审计的
历史状态。明确参数保持，未公开部分仍按编号假设；新TMAF测试RMSE24.0989，性能未复现。

更新：优先级1的四集数据窗口已独立重建，见[DATA_STRIDE50_REBUILD.md](DATA_STRIDE50_REBUILD.md)。
下文“未重建缓存/训练”仍指频谱与Qwen缓存及模型，并非新的原始数据窗口产物。

更新：优先级2/3已在FD001接入96维DKE+线性桥接、重训视觉线性projector与频谱融合。
见[SVLMA_ALIGNMENT_RESULTS.md](SVLMA_ALIGNMENT_RESULTS.md)。下表DKE未接入的描述是此前
审计状态；现在已按披露假设实现，但没有作者确认的原始桥接/训练目标，未重训最终TMAF。

依据用户提供的本地 PDF 第 III-B/C/D 节、第 IV-A/B/E 节和表 II。
**现有完整多模态结果均不能标为完全忠实复现主结果。** 本次修改参数和接口，不删除
旧产物，也未重新生成缓存/训练；旧指标不会因代码修改自动成为修正协议的新指标。

## 数据、时间建模与频谱

| 项目 | 论文依据 | 核对与处理 |
|---|---|---|
| FD001–004 | IV-A，表 I | 数据和时间基线四集完成，最终多模态仅FD001；最终覆盖未完成 |
| 删除1/5/6/10/16/18/19，14通道 | IV-A | 一致 |
| Min-Max | 式(20) | 算法一致；仅训练发动机拟合、测试超界不clip是安全假设，论文未明确拟合范围 |
| 窗口40 | IV-A，表II | 一致 |
| **样本滑窗stride50** | IV-A | 旧1不一致。新增全链路stride参数，新Qwen入口默认50；正文同时声称扩充样本，存在叙述矛盾，不可擅自把50改成1称原方法 |
| 验证stride | 未单独说明 | 新入口验证50是对统一预处理的解释假设；旧验证1保留 |
| RUL cap125 | IV-A | 一致；测试是否也截断官方标签未明确，当前train/test均截断需披露 |
| 窗口终点标签 | 未公开完整映射 | 当前终点RUL是复现假设 |
| 划分与few-shot单位 | 未给完整验证协议，图4比例5/10/20/50/100 | 发动机80/20、seed42、按发动机抽样、少样本重拟合scaler都是安全假设；全套少样本未完成 |
| 测试窗口/短轨迹 | 未公开 | 每发动机最后窗口、左侧复制首值是复现假设 |
| RMSE/Score | 式(21)/(22) | 实现公式一致：指数分母13/10 |
| 预测clip[0,125] | 未公开 | 主表用clip，另存raw；不能把clip协议称为论文明文 |
| MAE/MAPE | 图4 | MAE有，完整少样本MAPE及报告未完成 |
| Patch4/stride1/dim64/head1 | III-B，表II | 一致；样本stride50与Patch stride1不同 |
| 右侧复制padding，N=(L-P)/S+2 | III-B | 一致，40/4/1对应38patches |
| 可学习位置编码+Transformer | 式(1)/(2) | 一致；2层、FFN256、GELU、pre-norm、dropout0.1为假设 |
| 初始时间单模态训练 | III-B末段 | 先训练时间基线符合描述；预训练日程和checkpoint选择未公开 |
| RP/STFT/CWT、114×114 | III-C，表II | 一致 |
| RP联合多通道 | III-C，式(6) | 使用14维时刻向量；窗口二次缩放、距离/sqrt14、阈值0.25为假设，延迟嵌入未公开 |
| STFT/CWT分通道后可学习融合 | III-C | 两组softmax sensor logits可学习，缓存来源是CNN训练checkpoint；训练阶段及softmax形式是复现假设，不是固定均值 |
| RP最近邻、谱图Min-Max/双线性、不裁剪 | III-C | 一致 |
| STFT/CWT细节 | 未公开具体参数 | FFT16/hop4/Hann、8尺度、实值Morlet近似、kernel31、log1p均是补充假设 |
| uint8图像缓存 | 未公开量化 | 工程简化，有精度损失，不可称无损等价 |

固定FD001划分，train/val/test：stride1为13407/3324/100；train/val stride50为309/76/100。
旧缓存与新样本协议不兼容，不能直接复用。

## MAE、DKE和Qwen

| 项目 | 论文依据 | 核对与处理 |
|---|---|---|
| 预训练MAE、视觉128 | III-C，表II | 128一致；本地小MAE的patch19/2层/4头、15轮/mask0.75/AdamW/lr0.001为假设，并非公开论文权重 |
| encoder来源 | III-C：pre-trained MAE | 旧使用监督微调temporal_mae；新默认mae_pretrain/encoder_state。是否最终微调视觉分支未明确 |
| **线性projector** | III-C：linearly maps | 旧两层MLP+GELU+LN有差异；新默认Linear(128,1024)，旧架构命名legacy_mlp。bias未明确 |
| projector训练 | 未公开目标/日程 | mean prompt embedding MSE、5轮、lr0.001是复现假设，不是论文方法 |
| 专家知识prompt | III-C DKE | 当前传感器统计prompt，没有原专家知识模板；内容、阈值、传感器选择均是假设 |
| **文本96维+位置编码** | 表II，式(11) | 旧直接用Qwen1024维原生embedding，并未实现96维DKE。已新增独立96维DomainKnowledgeEmbedding模块，但尚未接入最终Qwen管线 |
| 96→Qwen兼容 | 式(13)拼接 | 论文未说明96维文本如何桥接到原生Qwen宽度及如何训练；这是未决问题，不能把任意新增桥接层称为论文方法 |
| **最大文本长度512** | 表II | 新默认512，旧128属于简化配置 |
| LLM输出截取 | 未规定只取31文本输出 | 新默认全部最多512文本输出+1视觉，形状[513,1024]；512是否含视觉未明确，按文本长度解释是假设 |
| Qwen版本/大小 | III-C只称Qwen，表III的Qwen3-0.6B是One Fits All对照 | 主模型型号未明确，采用Qwen3-0.6B是资源假设，不能从对照型号推断主模型 |
| 冻结Qwen/视觉、BF16 | 未公开冻结策略/精度 | 工程假设；冻结缓存无法用于LLM/projector端到端微调 |

新默认修正了明确的线性与长度，不代表已经解决DKE、权重和训练目标问题。

## TMAF、训练与实验报告

| 项目 | 论文依据 | 核对与处理 |
|---|---|---|
| temporal Query、LLM Key/Value、scaled attention | 式(14)–(17) | 线性投影、sqrt64一致 |
| global context沿时间复制 | III-D明文 | 新增global_broadcast解释；masked mean聚合仍是假设。旧token K/V是另一个解释，不是字面实现 |
| 选择性检索 vs 复制全局向量 | III-D内部冲突 | 所有Key相同则softmax沿Key轴均匀，Query不能选择细节；测试确认38×38权重恒为1/38。保留tokens与global_broadcast两配置，不能声称某一解释由论文唯一确定 |
| Key64、Output32 | 表II | 当前一致；正文称K/V都为N×Dk与Value32有矛盾，采用表II数值并披露 |
| concat线性融合门 | 式(18)/(19) | 无sigmoid一致；输出64是接口假设 |
| residual | 正文称residual，公式concat linear | 按公式线性融合，没有额外+FTS，不私自添加残差 |
| 回归头与pooling | 未公开完整结构 | mean pool、64→512→1、GELU是假设；表II明确512/dropout0.5 |
| batch128/lr0.002/30轮 | 表II | 原TMAF默认一致；Adam/MSE、验证最优选择是补充假设 |
| 冻结5轮、时间lr0.0002 | 论文未给 | **额外改进实验，不是论文方法**；默认freeze0、统一0.002，不能用12.5539作忠实复现主结果 |
| 模态消融 | IV-E删除输入模态 | 当前只mask输出token，文本已见视觉；不是严格输入模态消融。shuffled为额外负对照 |
| 生命周期曲线 | 图3 | 当前SVG是100台发动机终点折线，不是单发动机全寿命预测；图3未复现 |
| 最终4集/few-shot | IV-C/D | 最终模型四集、完整few-shot与输入模态消融均未完成 |

## 本次代码修正与证据

1. 新projector默认线性；旧MLP显式legacy；旧checkpoint与缓存原样保留。
2. 新Qwen入口默认512文本并保留全部输出，独立qwen_audited目录，默认加载预训练MAE。
3. 96维DKE独立模块落地，明确其尚未接入及桥接缺失，不伪称原1024维等于96维。
4. 数据/谱图缓存/Qwen/TMAF暴露train/val stride接口，可按50字面设置。
5. TMAF增加global_broadcast，协议输出目录隔离；manifest/result/checkpoint注明部分对齐与假设。
6. configs/paper_literal.json记录明确参数和未决项；它是审计事实表，**不是完整可执行配置**。
7. 测试24项通过，包含线性projector、96维DKE、512文本缓存mask、global broadcast均匀注意力。

RTX4060Ti实测：线性前缀+512文本的Qwen输出[1,513,1024]，独立DKE输出[1,512,96]，
global_broadcast融合输出[1,38,64]，attention与1/38差值为0。该检查使用原生Qwen
文本embedding，只是模块兼容性验证，**没有证明96维DKE已经接入**，也没有生成新实验指标。

## 现有结果归类及后续

temporal/CNN/ViT/MAE/Qwen-cache/TMAF/ablation/stability均归“假设配置实验”；
staged的12.5539归“额外训练策略实验”。统一lr一致不等于整个链路忠实。
论文主表目标FD001–004 RMSE=12.45/14.22/11.97/15.94，Score=233.40/929.81/338.30/1715.11。
数值接近不能证明方法等价。

本次不重建缓存/重训最终模型。优先明确96维文本兼容、MAE权重来源、projector训练、
global broadcast解释，并为每个未公开选择编号。作者实现/说明不可获得时，只能称
“明确参数对齐+假设完整披露”的复现，不是100%原样。明确前提后再生成stride50新频谱
和新Qwen缓存、统一0.002训练30轮；不要把旧缓存与新协议混用。
