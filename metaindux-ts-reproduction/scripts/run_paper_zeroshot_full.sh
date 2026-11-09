#!/usr/bin/env bash
# Table V zero-shot generation and fixed-split evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1

.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: run in a GPU-enabled host session"'
.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_zeroshot_full.yaml --skip-completed

for dataset in fd001 fd002 fd003 fd004; do
  .venv/bin/python scripts/evaluate_fd002_stability.py \
    --config "configs/evaluation_${dataset}_zeroshot.yaml"
done

.venv/bin/python analysis/audit/summarize_zeroshot.py
