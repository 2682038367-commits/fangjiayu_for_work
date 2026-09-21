# 有序工况序列：operating_sequence_v2

仅扩充 prompt 的工况序列表示，模型、时间/视觉分支、全局 Min-Max、窗口40、
样本步长50、Patch步长1、RUL/125、global_broadcast、优化器与训练日程保持不变。
通用知识、传感器摘要、组合占比、切换次数、末时刻原始设置均保留。

为满足512-token预算，使用当前窗口的组合字典和有序代码，例如：

```text
A=(0,0,100):20/40; B=(10,0.25,100):20/40.
Ordered observed-cycle codes, oldest to newest: A B A B ...
```

代码序列含每个实际观测周期，按时间先后排列；可以通过字典恢复各周期的三维
**舍入后**设置。沿用旧版本 setting1/2/3 的0/2/0位精度，不额外引入精度变化，
不声称保留原始设置的全部小数微扰。字母仅为本窗口字典键，不是跨窗口统一工况ID。
序列位置与输入传感器行的位置关系明确写入文本。短轨迹不伪造补齐行的工况观测。

具体序列化方式属于论文未公开的复现假设。它保留输入工况顺序，但没有改变
Qwen token 平均聚合，因此不等于已实现逐时间步/逐patch跨模态对齐。

## 验证

使用本地 Qwen tokenizer 对四子集 train/val/test 的共3726个窗口预检：

| 子集 | 最短 tokens | 最长 tokens |
|---|---:|---:|
| FD001 | 318 | 332 |
| FD002 | 387 | 427 |
| FD003 | 326 | 332 |
| FD004 | 401 | 427 |

全部低于512。对齐和缓存入口仍在截断前检查完整长度，超限直接报错，不静默删周期。
48项单元测试通过，包括40周期完整顺序和短轨迹位置检查。

## 运行

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/run_global_knowledge_tmaf.py
```

默认FD002、FD004，GPU重新训练对齐、重建缓存，再训练完整TMAF；来源匹配的
MAE和时间权重仍复用。新产物在 `artifacts/global_knowledge_sequence_v2/<FDxxx>/seed42/`，
旧 `global_knowledge_v1` 原样保留，不混用。
四集运行可加 `--datasets FD001 FD002 FD003 FD004`。
本次未启动训练，尚无新性能结论。
