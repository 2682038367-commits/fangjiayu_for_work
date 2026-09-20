#!/usr/bin/env python3
"""Run FD001 valid-patch-mean dcomp-8 Full Fusion for ten fixed seeds."""

from __future__ import annotations

import run_fd002_seed_sweep as sweep


sweep.DATASET = "FD001"
sweep.TRAIN_ENGINES = 80
sweep.VALIDATION_ENGINES = 20
sweep.TEST_ENGINES = 100
sweep.NUM_REGIMES = 1
sweep.SEED42_ELAPSED_SECONDS = 4.275
sweep.CACHE_DIR = (
    sweep.PROJECT_ROOT / "artifacts" / "FD001" / "representation_ablation" / "valid_patch_mean"
)
sweep.CACHE_MANIFEST = sweep.CACHE_DIR.parent / "manifest.json"
sweep.OUTPUT_DIR = sweep.CACHE_DIR
sweep.RESULTS_JSON = sweep.CACHE_DIR / "seed_results.json"
sweep.RESULTS_CSV = sweep.CACHE_DIR / "seed_results.csv"
sweep.REPORT_PATH = sweep.CACHE_DIR / "TEN_SEED_REPORT.md"


if __name__ == "__main__":
    sweep.main()
