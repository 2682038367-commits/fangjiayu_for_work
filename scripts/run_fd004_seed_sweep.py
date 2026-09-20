#!/usr/bin/env python3
"""Run the fixed FD004 valid-patch-mean Full Fusion configuration for ten seeds."""

from __future__ import annotations

from pathlib import Path

import run_fd002_seed_sweep as sweep


sweep.DATASET = "FD004"
sweep.TRAIN_ENGINES = 199
sweep.VALIDATION_ENGINES = 50
sweep.TEST_ENGINES = 248
sweep.NUM_REGIMES = 6
sweep.CACHE_DIR = sweep.PROJECT_ROOT / "artifacts" / sweep.DATASET
sweep.CACHE_MANIFEST = sweep.CACHE_DIR / "manifest.json"
sweep.OUTPUT_DIR = sweep.CACHE_DIR
sweep.RESULTS_JSON = sweep.CACHE_DIR / "seed_results.json"
sweep.RESULTS_CSV = sweep.CACHE_DIR / "seed_results.csv"
sweep.REPORT_PATH = sweep.CACHE_DIR / "TEN_SEED_REPORT.md"


if __name__ == "__main__":
    sweep.main()
