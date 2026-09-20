#!/usr/bin/env python3
"""Run the three cached-token TMAF ablations not already covered by full TMAF."""

from __future__ import annotations

from pathlib import Path

from ts_mllm.training_tmaf import parse_args, train


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for mode in ("visual_only", "text_only", "shuffled"):
        args = parse_args()
        args.token_mode = mode
        args.output_dir = project_root / "artifacts" / "tmaf_ablation" / mode / "FD001" / "seed42"
        print(f"\n=== TMAF ablation: {mode} ===", flush=True)
        train(args)


if __name__ == "__main__":
    main()
