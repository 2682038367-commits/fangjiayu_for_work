#!/usr/bin/env bash
# Run from the dedicated project environment; resume completed artifacts only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1
.venv/bin/python scripts/run_experiments.py \
  --config configs/binary_ste_fd002_all_seeds.yaml --skip-completed --reuse-generated
.venv/bin/python scripts/evaluate_fd002_stability.py \
  --config configs/evaluation_fd002_binary_ste_all_seeds.yaml
