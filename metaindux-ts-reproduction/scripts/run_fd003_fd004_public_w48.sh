#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1

.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fd003_fd004_w48_public_code.yaml --skip-completed
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd003_w48_public_code.yaml
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd004_w48_public_code.yaml
