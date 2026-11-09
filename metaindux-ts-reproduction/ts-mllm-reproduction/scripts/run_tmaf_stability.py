#!/usr/bin/env python3
"""Repeat full TMAF with fixed data/cache/temporal weights and report sample SD."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ts_mllm.training_tmaf import parse_args, train


def main() -> None:
    defaults = parse_args()
    root = Path(__file__).resolve().parents[1]
    results = []
    for seed in (42, 52, 62):
        args = copy.copy(defaults)
        args.seed = seed
        args.split_seed = 42
        args.token_mode = "full"
        args.output_dir = root / "artifacts/tmaf/FD001" / f"seed{seed}"
        result_path = args.output_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not (
                result["seed"] == seed
                and result["split_seed"] == 42
                and result["epochs"] == args.epochs
                and result["batch_size"] == args.batch_size
                and result["learning_rate"] == args.learning_rate
                and result.get("token_mode", "full") == "full"
                and Path(result["qwen_cache"]).resolve() == args.cache_dir.resolve()
                and Path(result["temporal_checkpoint"]).resolve() == args.temporal_checkpoint.resolve()
            ):
                raise ValueError(f"incompatible existing result: {result_path}")
            print(f"reuse seed={seed}: {result_path}", flush=True)
        else:
            print(f"train stability seed={seed}", flush=True)
            result = train(args)
        results.append(result)
    summary = {
        "seeds": [42, 52, 62],
        "split_seed": 42,
        "std_convention": "sample standard deviation (ddof=1)",
        "cache_and_temporal_checkpoint_fixed": True,
        "runs": [
            {"seed": r["seed"], "best_epoch": r["best_epoch"], "test": r["test"]}
            for r in results
        ],
        "aggregate": {
            key: {
                "mean": float(np.mean([r["test"][key] for r in results])),
                "std": float(np.std([r["test"][key] for r in results], ddof=1)),
                "min": float(min(r["test"][key] for r in results)),
                "max": float(max(r["test"][key] for r in results)),
            }
            for key in ("rmse", "mae", "score")
        },
    }
    output_dir = root / "artifacts/tmaf_stability/FD001"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
