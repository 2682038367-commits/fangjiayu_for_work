#!/usr/bin/env python3
"""Run FD001 mean-patch dcomp-8 Only Deep for ten fixed seeds."""

from __future__ import annotations

import run_fd001_ablation as ablation


ablation.DATASET = "FD001"
ablation.TEST_ENGINES = 100
ablation.NUM_REGIMES = 1
ablation.CACHE_DIR = (
    ablation.PROJECT_ROOT / "artifacts" / "FD001" / "representation_ablation" / "valid_patch_mean"
)
ablation.OUTPUT_DIR = ablation.CACHE_DIR / "only_deep_ablation"
ablation.COMPRESSION_DIMS = [8]
ablation.CONFIGS = [("fusion", 8), ("deep_only", 8)]
ablation.CSV_PATH = ablation.OUTPUT_DIR / "only_deep_runs.csv"
ablation.JSON_PATH = ablation.OUTPUT_DIR / "only_deep_summary.json"
ablation.REPORT_PATH = ablation.OUTPUT_DIR / "ONLY_DEEP_REPORT.md"


if __name__ == "__main__":
    ablation.main()
