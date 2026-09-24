# 文件清单（metaindux_handoff_20260922）

> **历史索引，禁止据此判定文件权威性或执行脚本（2026-09-24）**。其中对
> `protocol_gap_final.md`、`fd004_protocol_gap.md` 与 `protocol_gap_decomposition.py`
> 的“协议层”描述已经撤回。当前权威入口仅为
> `docs/conclusions/final/REPRODUCTION_AUDIT.md`、`results/frozen_numbers.json` 和
> `analysis/audit/audit_checklist.sh`。

状态标记：**AUTH** = 权威产物，结论由它支撑，不可删改 · **TOOL** = 工具脚本 · **DOC** = 文档 · **FIG** = 图表 · **ARCH** = 留档勿用

---

## 00 / 02 顶层

| 文件 | 状态 | 用途 |
|---|---|---|
| `00_START_HERE.md` | DOC | **接手 Agent 的任务书**，必读第 1。背景、路径、红线、工作顺序 |
| `MANIFEST.md` | DOC | 本文件 |
| `02_frozen_numbers/frozen_numbers.json` | AUTH | **数字唯一权威源**，10 个顶层键，全部由 `gen_frozen_numbers.py` 现场算出并断言自校验 |

---

## 01_conclusions —— 结论文档（12）

| 文件 | 状态 | 用途 | 与其他的关系 |
|---|---|---|---|
| `REPRODUCTION_AUDIT.md` | AUTH | **总纲**：§0 结论 / §1 目录地图 / §2 代码链路（文件:行号）/ §3 三层核对清单 / §4 已知异常 / §5 局限 / §6 FD004 下一步 | 读这一份就能上手；其余是它的支撑材料 |
| `protocol_gap_final.md` | AUTH | 三数据集协议层判别实验：显著性、异质性 Q/I²、FD002↔FD004 同构 Welch 对照、Provenance 核查、冻结数字表 | 归因分解的主报告 |
| `fd002_seven_seed_conclusion.md` | AUTH | 七种子配对检验：主结果、三条稳健性证据、DS 单点抖动、seed 53 溯源与离散度、功效分析、英文论文措辞 | 两臂对照的主报告 |
| `fd004_protocol_gap.md` | DOC | FD004 +1.329 的悬案分析与判别实验设计（三个候选解释 + 三分支判读表） | 设计文档；结论已被 `protocol_gap_final.md` 收口 |
| `fd003_fd004_baseline_attribution.md` | DOC | FD003/FD004 真实对照的早期归因 | 早期版本，数字以 JSON 为准 |
| `fd003_fd004_statistical_analysis.md` | DOC | FD003/FD004 的统计分析过程 | 同上 |
| `theta_analysis_report.md` | DOC | 历史 θ 参数分析；不等同于实测丢弃率 | 已被 mask-drop 口径取代 |
| `theta_rmse_report.md` | DOC | 历史 θ 与 RMSE 分析 | 不得用于跨模式丢弃率结论 |
| `fd002_real_baseline_howto.md` | DOC | FD002 真实对照三步操作说明（check / run / extract） | 配 `03_scripts/run_fd002_real_baseline_v2.sh` |
| `fd002_fd001_fixed_threshold_plan.md` | DOC | FD002/FD001 固定阈值补跑计划 | 计划文档，已执行完毕 |
| `fd003_baseline_local_checklist.md` | DOC | FD003 本地对照核对清单 | 早期清单 |
| `where_we_are_now.md` | DOC | 早期进度快照 | 仅作历史参考，状态已被总纲取代 |

---

## 03_scripts —— 脚本（22，放到项目根目录运行）

> 全部使用相对路径，工作目录须为项目根目录。Python 用 `.venv/bin/python`。

### 核对 / 复算（首选入口）

| 文件 | 状态 | 用途 |
|---|---|---|
| `audit_checklist.sh` | TOOL | **一键核对**，43 项 A/B/C 三层检查，只读不写，输出 PASS/FAIL/INFO 计数 |
| `gen_frozen_numbers.py` | TOOL | **生成 + 自校验冻结数字 JSON**；`--check` 只校验不写文件，退出码 0 = 全部一致 |
| `verify_seven_seed_eval.py` | TOOL | 七种子评估自检 v2：区分「没跑完」与「真错」，退出码 0/1/2 |
| `verify_fd002_real_fresh.sh` | TOOL | FD002 真实对照的时序验证：比对新旧 csv 行、查日志、算 sha256 |
| `ds_drift_check.py` | TOOL | 判别分数抖动体检：逐格子新旧对比 + 相对偏差百分比 |

### 分析

| 文件 | 状态 | 用途 |
|---|---|---|
| `protocol_gap_decomposition.py` | TOOL | 归因恒等式分解：协议层 + 生成端，含单样本 t 检验、两臂敏感性 |
| `final_seven_seed_test.py` | TOOL | 七对配对检验 + 35 次 leave-one-cell-out + 事后检出功效，出图与报告 |
| `theta_analysis.py` | TOOL | θ 机制分析 |
| `theta_rmse_analysis.py` | TOOL | θ 与 RMSE 关系分析，出 `theta_rmse.png` |
| `read_theta_from_ckpt.py` | TOOL | 从 ckpt 直接读 θ 值（验证固定臂确实 θ=0.25） |
| `threshold_ste_probe.py` | TOOL | STE 能量阈值梯度探针（不是实测丢弃比例） |
| `collect_seed_metrics.py` | TOOL | 汇总各 seed 指标 |
| `fd34_analysis.py` | TOOL | FD003/FD004 分析 |

### 训练 / 评估驱动

| 文件 | 状态 | 用途 |
|---|---|---|
| `backfill_fixed_arm.py` | TOOL | 固定臂缺失种子补跑（含 `--verify` 校验 θ） |
| `start_backfill.sh` | TOOL | 启动补跑（并行，~37 min/种子） |
| `check_backfill.sh` | TOOL | 补跑进度体检 |
| `check_now.sh` | TOOL | 当前状态快查 |
| `run_eval_seven_seeds.sh` | TOOL | 七种子评估启动脚本（~45 min） |
| `resume_eval.sh` | TOOL | 评估断点补跑（先备份现有行） |
| `run_fd002_real_baseline_v2.sh` | TOOL | FD002 真实对照三步（check/run/extract），**v2 为当前版本** |
| `run_fd002_real_baseline.sh` | ARCH | 同上，**v1，已被 v2 取代**，留档 |

---

## 04_configs —— 配置（2）

| 文件 | 状态 | 用途 |
|---|---|---|
| `evaluation_fd002_frequency_threshold_seven_seeds.yaml` | AUTH | 七种子评估配置（generation_seeds 7 个、evaluator_seeds 5 个） |
| `frequency_threshold_validation_fd002_remaining_seeds.yaml` | DOC | 补跑剩余种子的配置 |

> 注意：公开臂评估用的 `configs/evaluation_fd002_public_arm_seven_seeds.yaml` 在**目标机器上生成**，本包未含，需从项目目录取。

---

## 05_figures —— 图表（5）

| 文件 | 状态 | 说明 |
|---|---|---|
| `theta_rmse.png` | FIG | 历史 θ 与 RMSE 图；能量阈值不等于丢弃比例 |
| `theta_scatter.png` | FIG | θ 散点图（同上修正） |
| `theta_rmse_report.html` | FIG | 交互式 HTML 报告 |
| `theta_report.html` | FIG | 交互式 HTML 报告 |
| `theta_scatter_template.html` | DOC | 出图模板 |

---

## 06_archive_superseded —— 留档勿用（11）

**内容已全部被 01–05 目录取代且更全，不要解压使用。** 存在这些包只因分次交付时产生过版本混乱（拆出过缺文件的旧包），保留仅为留档追溯。

`metaindux_all_20260921.tar.gz` · `eval_stage_files.tar.gz` · `final_stage_kit_v2.tar.gz` · `final_test_kit.tar.gz` · `fd002_real_baseline_kit.tar.gz` · `fd002_real_baseline_kit_v2.tar.gz` · `fd004_protocol_gap_kit.tar.gz` · `protocol_gap_final_kit.tar.gz` · `reproduction_audit_kit.tar.gz` · `check_backfill_only.tar.gz` · `check_now.tar.gz`
