#!/usr/bin/env python3
"""Run FD004 mean-patch dcomp-8 Only Deep for ten fixed seeds."""

from __future__ import annotations

import run_fd001_ablation as ablation


ablation.DATASET = "FD004"
ablation.TEST_ENGINES = 248
ablation.NUM_REGIMES = 6
ablation.CACHE_DIR = ablation.PROJECT_ROOT / "artifacts" / "FD004"
ablation.OUTPUT_DIR = ablation.CACHE_DIR / "only_deep_ablation"
ablation.COMPRESSION_DIMS = [8]
ablation.CONFIGS = [("fusion", 8), ("deep_only", 8)]
ablation.CSV_PATH = ablation.OUTPUT_DIR / "only_deep_runs.csv"
ablation.JSON_PATH = ablation.OUTPUT_DIR / "only_deep_summary.json"
ablation.REPORT_PATH = ablation.OUTPUT_DIR / "ONLY_DEEP_REPORT.md"


if __name__ == "__main__":
    ablation.main()
