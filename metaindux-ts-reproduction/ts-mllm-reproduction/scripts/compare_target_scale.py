#!/usr/bin/env python3
import argparse
from pathlib import Path

from ts_mllm.target_scale import run_experiment


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/target_scale/FD001/seed42_verified")
    args = parser.parse_args()
    run_experiment(root, args.output_root)
