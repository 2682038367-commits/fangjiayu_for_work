#!/usr/bin/env bash
# FD001 / w48 formal public-code arm: 5 generation seeds × 5 evaluator seeds,
# followed by the matching local real-data baseline.  This script does not alter
# any published/code-specified model setting.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1

.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fd001_w48_public_code.yaml --skip-completed
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd001_w48_public_code.yaml
.venv/bin/python scripts/evaluate_real_data_baseline.py --datasets FD001
