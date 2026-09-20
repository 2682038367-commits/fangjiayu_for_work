#!/usr/bin/env python3
"""Run the FD002 mean-patch fusion and compression-dimension ablations."""

from __future__ import annotations

import run_fd001_ablation as ablation


ablation.DATASET = "FD002"
ablation.TEST_ENGINES = 259
ablation.NUM_REGIMES = 6
ablation.CACHE_DIR = ablation.PROJECT_ROOT / "artifacts" / "FD002"
ablation.OUTPUT_DIR = ablation.CACHE_DIR / "ablations"
ablation.CSV_PATH = ablation.OUTPUT_DIR / "ablation_runs.csv"
ablation.JSON_PATH = ablation.OUTPUT_DIR / "ablation_summary.json"
ablation.REPORT_PATH = ablation.OUTPUT_DIR / "ABLATION_REPORT.md"


if __name__ == "__main__":
    ablation.main()
