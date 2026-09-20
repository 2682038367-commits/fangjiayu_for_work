#!/usr/bin/env python3
"""Rebuild all four subsets with the paper's explicit window/stride values."""

import argparse
import json
from pathlib import Path

from ts_mllm.rebuilt_data import rebuild_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root.parent / "data/CMAPSSData")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/data_window40_stride50/split_seed42")
    args = parser.parse_args()
    reports = {}
    for name in ("FD001", "FD002", "FD003", "FD004"):
        report = rebuild_dataset(args.data_dir, args.output_root / name, name, seed=42)
        reports[name] = {split: details["count"] for split, details in report["splits"].items()}
        print(f"{name}: {reports[name]} PASS", flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
