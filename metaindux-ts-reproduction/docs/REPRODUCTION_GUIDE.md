# MetaIndux-TS 完整复现步骤

本指南对应项目冻结的**公开代码主线**：C-MAPSS FD001–FD004、窗口长度 48。
它保留论文和作者公开代码已明确的设置；公开代码未明确的评估随机性用固定
五个生成种子与五个评估器种子记录。Soft、STE、固定阈值和学习率实验均为
历史诊断，不能替代本指南中的主结果。

## 1. 进入项目与建立环境

```bash
cd /home/cw_boe/projects/rul/metaindux-ts-reproduction
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
```

已跑过的环境以 `requirements-lock.txt` 为准。主线使用 PyTorch 2.1.2 +
CUDA 12.1；训练必须在可见 NVIDIA GPU 的普通宿主机终端中执行：

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果第二条为 `False`，先修复容器/沙箱的 GPU 设备透传，不要改为 CPU 正式训练。

## 2. 准备并验证 C-MAPSS 数据

将 C-MAPSS 的 `train_FD00*.txt`、`test_FD00*.txt`、`RUL_FD00*.txt` 放在一个
稳定的外部目录。FD001 文件由作者快照的 `upstream/data/` 提供；FD002–FD004
需要链接进 `upstream/data/`。项目还保留一个指向同一数据目录的
`data/raw/CMAPSSData` 链接，便于记录数据来源。

例如（把 `/absolute/path/CMAPSSData` 替换为真实绝对路径）：

```bash
ln -s /absolute/path/CMAPSSData data/raw/CMAPSSData
for ds in FD002 FD003 FD004; do
  for prefix in train test RUL; do
    ln -s /absolute/path/CMAPSSData/${prefix}_${ds}.txt upstream/data/${prefix}_${ds}.txt
  done
done
```

不要把原始数据提交到 Git。先只验证数据处理；预期训练特征最后一维为 14，
测试特征最后一维也为 14：

```bash
cd upstream
../.venv/bin/python -c "
from data.CMAPSSDataset import CMAPSSDataset
d = CMAPSSDataset('FD001', 48, [1000])
x = d.get_feature_slice(d.get_train_data())
y = d.get_label_slice(d.get_train_data())
xt, yt = d.get_last_data_slice(d.get_test_data())
print(x.shape, y.shape, xt.shape, yt.shape)
"
cd ..
```

数据实现固定使用 14 个传感器、训练集拟合 MinMaxScaler、窗口末端 RUL 标签、
RUL 最大截断 125，测试中每台发动机仅取最后一个窗口并在必要时首行前补齐。

## 3. 冒烟测试

```bash
.venv/bin/python scripts/smoke_test.py
```

该步骤验证数据加载、作者提供 checkpoint 和一次 GPU 前向传播；它不是正式训练。

## 4. 冻结的主线配置

| 项目 | 取值 | 来源/说明 |
|---|---|---|
| 模型 | `DiffUnet_fre` | 作者公开代码 |
| 输入 | 14 传感器，窗口 48 | 公开数据处理 |
| 扩散 | `T=1000`、linear beta、DDPM | 论文/公开配置 |
| 训练 | 70 epoch、Adam、lr=0.002、batch=256、warmup-cosine、grad clip=1.0 | 论文明确项与公开代码默认项 |
| 频率阈值 | `hard_random_quantile` | 公开代码主线；不改为 soft/STE/固定值 |
| 生成种子 | FD001/003/004：3、13、23、33、43；FD002：额外加入 53、63 | 预先记录的重复设计 |
| 评估 | 固定切分种子 20260915；评估器种子 3、13、23、33、43；35 epoch | 项目统一、可复算的未公开评估设置 |

## 5. 生成、采样与正式评估

所有命令均从项目根目录执行。先用 `--dry-run` 检查命令；正式运行可用
`--skip-completed` 从中断处恢复已完整生成的产物。

### FD001

```bash
bash scripts/run_fd001_public_w48.sh
```

该脚本依次执行 5 个生成种子训练/采样、25 个固定切分评估单元，以及 FD001
本地真实数据基线。配置为 `configs/paper_fd001_w48_public_code.yaml` 与
`configs/evaluation_fd001_w48_public_code.yaml`。

### FD002：公开代码主线

```bash
.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fd002.yaml --skip-completed
.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fd002_extra_seeds.yaml --skip-completed
TF_ENABLE_ONEDNN_OPTS=0 TF_DETERMINISTIC_OPS=1 \
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd002_public_arm_seven_seeds.yaml
.venv/bin/python scripts/evaluate_real_data_baseline.py --datasets FD002
```

第一份生成配置覆盖种子 3/13/23/33/43，第二份覆盖 53/63；正式结果共有
7 x 5 = 35 个评估单元。固定 θ=0.25 对照仅用于 FD002 消融，不能混入公开代码主线。

### FD003 与 FD004

```bash
bash scripts/run_fd003_fd004_public_w48.sh
.venv/bin/python scripts/evaluate_real_data_baseline.py --datasets FD003 FD004
```

该脚本进行两个数据集各 5 个生成种子的训练/采样与正式评估；每个数据集有
5 x 5 = 25 个评估单元。

## 6. 产物检查与结果冻结

正式结果只从 `results/*_evaluation_raw.csv`、对应 summary CSV 和本地真实基线
读取。训练 checkpoint、合成 `.npz`、日志和原始 C-MAPSS 数据均为本地大文件，
不上传 Git。

每次新增或重跑完成后，按顺序执行：

```bash
.venv/bin/python analysis/audit/gen_frozen_numbers.py
.venv/bin/python analysis/audit/gen_frozen_numbers.py --check
.venv/bin/python analysis/diagnostics/protocol_gap_decomposition.py
bash analysis/audit/audit_checklist.sh .
```

最后一条应报告 `FAIL 0`。最终可引用数字仅取自
`results/frozen_numbers.json` 的 `observable_comparisons`；面向报告的解释在
`docs/conclusions/final/REPRODUCTION_AUDIT.md`。

## 7. 结果解释边界

- 论文 Table I 报告的是“用生成数据训练”的预测 RMSE 点估计。
- 本项目的“正式复现差距”是本地生成数据 RMSE 减论文生成数据 RMSE，
  不是统计显著性检验。
- 本地生成数据惩罚是本地生成数据 RMSE 减本地真实数据 RMSE。
- 论文未报告真实数据训练基线，故不能估计论文与本地的协议差异，也不能计算
  论文自身的生成数据惩罚。
