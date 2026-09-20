# TS-MLLM 第二阶段：C-MAPSS 数据管线

该实现与现有 `rul_chronos` 代码相互独立。以下列表混合了论文明确设置与安全复现
假设，不能整体称为论文原始协议。完整区分见 [PAPER_ALIGNMENT_AUDIT.md](PAPER_ALIGNMENT_AUDIT.md)：

- 删除传感器 1、5、6、10、16、18、19，保留 14 个通道；
- RUL 使用上限为 125 的分段线性标签；
- 样本窗口形状为 `[40, 14]`；
- Min-Max 统计量仅使用实际参与训练的发动机拟合；
- 训练、验证和少样本采样均以发动机为单位；
- 验证集使用留出发动机的全部滑动窗口，避免只评估训练轨迹末端的零 RUL；
- 测试集每台发动机只取最后一个窗口，并与官方 RUL 文件逐台对应；不足 40
  个观测的短序列在左侧复制首个观测，确保不丢弃任何测试发动机。

论文同时报告窗口长度 40 和滑动步长 50，但又声称滑窗会扩充样本，二者存在矛盾。因此实现默认使用步长 1，并保留 `--train-stride 50` 作为核对实验。

## 数据审计

```bash
cd ts-mllm-reproduction
PYTHONPATH=src ../.venv/bin/python scripts/audit_ts_mllm_data.py
```

核对论文中的步长 50：

```bash
PYTHONPATH=src ../.venv/bin/python scripts/audit_ts_mllm_data.py --train-stride 50
```

审计会检查固定输入形状、训练/验证发动机隔离、RUL 取值范围及单调性，并输出四个子集的发动机数和窗口数。

## Python接口

```python
from ts_mllm.data import prepare_cmapss_data

data = prepare_cmapss_data(
    "../data/CMAPSSData",
    "FD001",
    seed=42,
    window_size=40,
    train_stride=1,
    rul_cap=125,
)

sample = data.train[0]
assert sample["x"].shape == (40, 14)
```

少样本实验通过 `few_shot_fraction` 配置，例如 `0.05`、`0.1`、`0.2` 和 `0.5`。归一化器会在选中的少样本训练发动机上重新拟合。
