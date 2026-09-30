#!/usr/bin/env bash
# Formal public-code w24 pipeline: 4 datasets x 5 generation seeds, followed
# by 4 x 25 fixed-split evaluator cells. Safe to rerun after interruption.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1

.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fd001_fd004_w24_public_code.yaml \
  --skip-completed

for dataset in fd001 fd002 fd003 fd004; do
  .venv/bin/python scripts/evaluate_fd002_stability.py \
    --config "configs/evaluation_${dataset}_w24_public_code.yaml"
done

echo "w24 public-code generation and formal evaluation complete"
