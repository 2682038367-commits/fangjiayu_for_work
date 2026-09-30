#!/usr/bin/env bash
# Full Table VI MetaIndux-TS few-shot pipeline. Safe to rerun.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_DETERMINISTIC_OPS=1

while systemctl --user is-active --quiet metaindux-fewshot-fd001.service; do
  sleep 30
done

.venv/bin/python scripts/run_experiments.py \
  --config configs/paper_fewshot_full.yaml \
  --skip-completed

for dataset in fd001 fd002 fd003 fd004; do
  .venv/bin/python scripts/evaluate_fd002_stability.py \
    --config "configs/evaluation_${dataset}_fewshot.yaml"
done

echo "full Table VI few-shot generation and evaluation complete"
