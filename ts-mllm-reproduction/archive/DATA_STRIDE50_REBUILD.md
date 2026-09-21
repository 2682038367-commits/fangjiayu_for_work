# 优先级1：窗口40、样本步长50数据重建

已独立导出FD001–FD004数据，路径为
`artifacts/data_window40_stride50/split_seed42/{FD001,FD002,FD003,FD004}/`。
不覆盖旧stride1的频谱、Qwen缓存或模型，也不声称它们已适配新数据。

| 子集 | Train窗口 | Validation窗口 | Test窗口 |
|---|---:|---:|---:|
| FD001 | 309 | 76 | 100 |
| FD002 | 812 | 195 | 259 |
| FD003 | 377 | 92 | 100 |
| FD004 | 938 | 220 | 248 |

## 明确设置

- 样本窗口40周期，14有效传感器，float32；
- 训练样本步长50；RUL上限125；
- Patch size4、Patch stride1未改变，后续时间分支仍得到38个patch；
- 每个窗口RUL对应窗口最后一个观测时刻。

## 保持并披露的复现假设

- 发动机80/20划分，split seed42，归一化仅训练发动机拟合；
- 验证集也使用样本步长50，作为统一滑窗预处理解释；
- 从row0开始，窗口起点0/50/100…，只保留完整窗口，不追加偏离stride的终点；
- 因此训练发动机最后一个选中窗口不一定到失效点，不要求最后RUL为0；
- 测试每台发动机仍只使用最后窗口，并与官方RUL逐台对应，标签截断125；
- 短测试轨迹左侧复制首观测，不因长度不足40而丢发动机。

## 导出和校验

每个split导出x、target、unit、cycle、start_row、left_padding六个npy文件。
另有scaler.json与manifest.json，记录发动机划分、协议、源数据与输出SHA256。
审计检查：有限值、[N,40,14]、RUL范围和单调性、每台发动机窗口计数、起点间隔50、
窗口终点标签、train/val发动机不相交、测试发动机与官方标签映射、落盘数据一致性。

没有训练模型，数据读取/归一化/文件导出在CPU完成；后续模型训练与特征提取继续用GPU。

## 入口及后续读取

```bash
PYTHONPATH=src ../.venv/bin/python scripts/rebuild_paper_data.py
```

脚本拒绝覆盖已存在的子集目录；需要复跑时指定新的--output-root。

```python
from ts_mllm.rebuilt_data import RebuiltWindowDataset

train = RebuiltWindowDataset(
    "artifacts/data_window40_stride50/split_seed42/FD001", "train"
)
assert train[0]["x"].shape == (40, 14)
```

加载器校验文件SHA256并保持导出样本顺序，供下一阶段频谱/Qwen缓存使用。
现有训练/频谱脚本尚未自动切换到此导出加载器，不应直接运行legacy默认命令。
