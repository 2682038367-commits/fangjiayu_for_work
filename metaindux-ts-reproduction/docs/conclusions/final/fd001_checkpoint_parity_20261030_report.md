# FD001两检查点实际条件审计（2026-10-30）

[Observed] 仅核对已有训练/采样记录、保存的标签、当前与已冻结源代码，并在CPU重建网络以严格加载权重；没有训练、采样、模型前向推理或新指标评价。

## 结论

[Observed] 未发现两边声明配置、数据任务或参数加载方式不一致。**选中权重的训练年龄不同**：两次完整训练均70轮，但同一最佳训练损失规则分别选择零基epoch63和14，即第64轮和第15轮。它们不是最终70轮权重。

[Code] **调用eval()不代表手写随机条件丢弃已关闭**：UNet在forward中直接使用默认cond_drop_prob=0.2，不检查self.training；条件CFG分支未显式覆盖为0。两个检查点采样均受此逻辑影响。已留存的公开代码也有同一行为。不能直接宣称它就是两检查点差距的原因。

| 核对项 | seed13 | seed23 | 判断 |
|---|---|---|---|
| 任务与窗口 | FD001全数据；15931×48×14 | 相同 | 记录一致 |
| 预处理与标签 | 全训练集MinMax；窗口末端RUL截断125 | 相同 | 现存标签逐元素一致 |
| 网络参数量 / 状态张量数 | 526851 / 194 | 526851 / 194 | 结构一致 |
| 完整训练轮数 | 70 | 70 | 两份epoch0–69完整记录 |
| 每轮批次 / 总更新次数 | 63 / 4410 | 63 / 4410 | 按循环和epoch记录重建 |
| 保存模型的轮次（从1计） | **64** | **15** | 不同 |
| 保存模型对应更新次数 | **4032** | **945** | 按保存epoch重建，不是等训练年龄 |
| 保存规则 | epoch>5后，训练扩散损失均值创新低即覆盖 | 相同 | 两边均为best-loss，均非final |
| 完整加载参数 | strict=True；missing/unexpected均为空 | 相同 | 194个张量逐元素匹配 |
| eval标志 | 210个子模块全部False | 相同 | 正确进入评价模式 |
| eval后条件随机丢弃 | 条件分支仍为0.2 | 相同 | 自定义逻辑不受eval控制 |

## 数据及执行配置的证据边界

[Observed] 两份per-run metadata的config字典完全相同；逐项CLI比较仅seed及输出文件路径不同。2026-09-24两份独立W&B离线metadata也记录相同命令。两份70轮学习率序列逐元素相同，框架/GPU版本记录一致。

[Observed] 当前重建的真实训练窗口及标签与2026-10-29冻结输入逐元素一致；原seed13/23生成输出标签也逐元素一致，当前训练文件和源代码哈希与前次冻结记录一致。**原训练时未分别保存两份处理后真实输入张量的哈希**，因此不能把事后重建表述成训练当时逐元素输入已独立封存验证。

[Observed] 检查点仅存模型state_dict，不含optimizer状态或逐步计数器。4410与4032/945均是由完整epoch记录、15931样本、batch256、drop_last默认False以及每批一次无条件optimizer.step重建的更新次数，不冒充直接读取的计数器。70轮实际完成记录不意味着所加载权重已经历70轮。

## 采样加载与评价模式

[Code] TrainCondition.sample先构建同结构模型，应用frequency_threshold=-1（不覆盖），strict默认True加载state_dict，再调用net_model.eval()；随后包装DDPM。原采样不修改已加载参数。六次采样记录的checkpoint哈希与现有文件一致。

[Observed] 本次CPU加载两checkpoint均无缺失/多余键；加载后194个状态张量逐元素与文件相同；210个子模块training标志均False。模型没有普通nn.Dropout模块，不能把手写条件随机掩码当作普通Dropout模块处理。

[Code] 路径：GaussianDiffusion.model_predictions → forward_with_cond_scale → 首次forward不显式传cond_drop_prob → 使用self.cond_drop_prob=0.2 → 对每个窗口随机把条件嵌入换成null_classes_emb。随后无条件分支显式cond_drop_prob=1，这是CFG正常构造无条件输出的分支。这里不是训练期间整批数值RUL置零，两者须分开。

[Code] 证据：`upstream/TrainCondition.py:265`（完整加载）、`:269`（eval）；`upstream/DiffusionFreeGuidence/Unet1D_fre.py:412`（CFG）、`:435`（默认丢弃率）、`:441`（随机条件掩码）；`upstream/GaussianDiffusion.py:207`（采样调用CFG）。留存公开代码：`docs/provenance/author_label_zeroing/Unet1D_fre.py.public.txt:404`和`:427`。

## 与种子关联的其他状态

[Observed] 两个存储频率分位参数分别为seed13：[0.298574,0.495990]，seed23：[0.074127,0.496258]。在相同网络构造下按各自训练种子初始化，得到的两个参数与保存值逐元素相同。这不是加载遗漏，也不是相同CLI配置保证这些随机初始化参数相同。

[Hypothesis] 此次结果支持“完整检查点状态关联的稳定差异”，尚不能拆分训练年龄、随机频率参数及其他已学习权重的贡献。条件丢弃是双方共有采样行为，并非已证实的组间不公平因素；也不能由本次静态审计推断关闭它一定提高判别或下游效果。

## 停止点及下一步

[Hypothesis] 先明确推理条件分支应保留还是禁用训练式随机条件丢弃。如果做验证，应单独固定检查点和采样设置，仅将条件分支cond_drop_prob从0.2改为0，保留CFG无条件分支=1；该对照需要单独授权，当前未执行。不要同时更换模型或训练轮次。

[Observed] 没有可复用的最终70轮生成器权重副本：现有训练代码只保存best-loss覆盖文件；本次只检查目标权重文件和已有保存逻辑，不重新训练、不替换严格复现或历史采样结果。不能把现有best权重当成最终权重进行“同轮”比较。

[Observed] 五段比较：当前为已有记录/参数审计 → 论文无对应审计指标 → 数值差距不适用 → 不新增种子或指标 → 严格复现结论不变。仅完成本次审计并停止；零标签方向仍暂停，EXT-002仍保留weight1。

[Observed] Run `diagnostic-fd001-checkpoint-parity-20261030`；机器结果 `results/diagnostics/fd001_checkpoint_parity_20261030/audit_findings.json`；快照 `runs/diagnostic-fd001-checkpoint-parity-20261030/run-manifest.json`。
