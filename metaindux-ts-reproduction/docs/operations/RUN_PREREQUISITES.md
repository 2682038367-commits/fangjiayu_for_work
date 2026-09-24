# 03_scripts 运行前提

1. **工作目录必须是项目根目录** `~/projects/rul/metaindux-ts-reproduction`。
   所有脚本用相对路径（`results/...`、`outputs/...`、`configs/...`），不做路径自发现。
   把本目录内容复制到项目根目录，或在项目根目录下软链。

2. **Python 用 `.venv/bin/python`**，不要用系统 python（缺 numpy/scipy/torch/tensorflow 等依赖）。
   纯统计脚本（不含 torch）用系统 python 也能跑，但为一致性建议统一用 venv。

3. **只读优先**。以下脚本不会写任何文件：
   `audit_checklist.sh`（默认）、`gen_frozen_numbers.py --check`、
   `verify_seven_seed_eval.py`、`verify_fd002_real_fresh.sh`、`ds_drift_check.py`。
   会写文件的只有 `gen_frozen_numbers.py`（写 JSON）、`final_seven_seed_test.py`（写图 + 报告）、
   以及 `run_*` / `resume_eval.sh` 这类驱动脚本——它们会备份后再写。

4. **推荐入口**：
   ```bash
   bash audit_checklist.sh                     # 43 项核对，先看这个
   .venv/bin/python gen_frozen_numbers.py --check   # 冻结数字复算一致性
   ```

5. **不建议在本轮运行**的（会占 GPU 30–45 分钟，且结论已冻结，无需重跑）：
   `run_eval_seven_seeds.sh`、`resume_eval.sh`、`run_fd002_real_baseline_v2.sh run`、
   `start_backfill.sh`。它们留在这里是为了可复现性，不是待办。
