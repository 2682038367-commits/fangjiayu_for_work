# FD001/FD003来源核对与最小修复

本次只核对文件及准备GPU修复入口，尚未启动训练。

| 项目 | FD001原结果 | FD003 |
|---|---|---|
| 数据 | window40/样本stride50 | window40/样本stride50 |
| MAE来源 | `mae_pretrain/FD001/seed42`，旧stride1谱图缓存 | `mae_pretrain_audited/FD003/stride50_seed42`，新stride50窗口 |
| MAE谱图流程 | 旧CNN训练后融合，uint8缓存 | 在线float32，均匀初始化传感器融合 |
| MAE checkpoint数据manifest哈希 | 未记录 | 与当前FD003数据一致 |
| MAE架构 | 114×114、patch19、dim128、4head、2层 | 相同 |
| MAE日程 | 15epochs、mask0.75 | 15epochs、mask0.75 |
| 对齐/缓存/TMAF数据哈希 | 匹配 | 匹配 |
| 缓存所指MAE/对齐权重哈希 | 匹配 | 匹配 |
| Qwen缓存所有文件checksum | 通过 | 通过 |
| float32谱图数据/对齐来源及所有checksum | 通过 | 通过 |
| TMAF所指缓存、时间权重哈希 | 匹配 | 匹配 |

FD001旧MAE谱图cache manifest训练/验证数量为13407/3324；新窗口数据为309/76。
旧MAE虽未绑定manifest哈希，但其result指定旧cache路径，谱图manifest及checkpoint架构已交叉核对。
因此问题是**MAE预训练输入协议不一致**，不是下游错配缓存文件。
FD003 MAE checkpoint明确绑定当前data manifest，未发现需重跑的来源差异。

## 只修复FD001受影响链路

新MAE按FD003的在线float32/均匀初始融合/stride50流程预训练，再重训对齐，
重建频谱和完整Qwen缓存，最后重训完整TMAF。不是仅改变stride的单变量性能实验：
谱图精度和MAE阶段融合来源也同步统一。

原始数据、train/val划分、全局Min-Max和B时间权重不变；不重训时间基线。
原统计prompt（legacy）、mean广播、窗口40、sample stride50、Patch stride1、
RUL/125、默认初始化、30epochs、batch128、统一lr0.002、无冻结、seed42不变。
MAE15epochs/AdamW lr0.001/mask0.75沿用其他子集的预训练假设，并非论文公开日程。
工况归一化FD002/FD004方案A及FD003全部产物不动。

```bash
cd /home/cw_boe/projects/rul/ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/repair_fd001_mae_protocol.py
```

各训练/缓存阶段要求GPU，最后独立GPU核验指标及100行测试CSV。
可加`--dry-run`只检查命令。已完成阶段按来源复用，检测到部分训练权重时拒绝覆盖。
新输出在`artifacts/fd001_mae_stride50_repair/seed42/{mae,alignment,qwen,spectrum,tmaf}/`。
旧FD001 RMSE15.2218等结果保留，但不得继续称为统一MAE协议的主结果。
修复不保证RMSE改善；目的是统一来源，而不是挑选更好的测试指标。
本次52项测试通过、修复命令dry-run通过，新结果待终端运行。
