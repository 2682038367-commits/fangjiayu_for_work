# C-MAPSS 数据

默认情况下，审计脚本会复用工作区中的数据：

```text
../../data/CMAPSSData/
```

也可以把 NASA C-MAPSS 文件放入本目录的 `CMAPSSData/`，然后运行：

```bash
python scripts/audit_ts_mllm_data.py --data-dir data/CMAPSSData
```

所需文件为 `train_FD001.txt`、`test_FD001.txt`、`RUL_FD001.txt`，以及
FD002、FD003、FD004 对应的同名文件。
