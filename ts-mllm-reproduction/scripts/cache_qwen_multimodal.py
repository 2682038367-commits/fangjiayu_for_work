#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ts_mllm.qwen_cache import run


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train MAE-to-Qwen projector and cache frozen Qwen multimodal tokens."
    )
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--prompt-profile", choices=["legacy", "operating_sequence_v2", "condition_operating_sequence_v1", "figure12_v1"], default="legacy")
    parser.add_argument("--input-modality", choices=["multimodal", "text_only", "visual_only"], default="multimodal",
                        help="Actual Qwen input ablation; do not use cache-output masking as a substitute.")
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument(
        "--spectrum-cache-dir", type=Path,
        default=project_root / "artifacts/spectrum_cache/FD001/split_seed42"
    )
    parser.add_argument(
        "--vision-checkpoint", type=Path,
        default=project_root / "artifacts/mae_pretrain/FD001/seed42/best.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "artifacts/qwen_audited/FD001/seed42"
    )
    parser.add_argument("--projector-checkpoint", type=Path)
    parser.add_argument(
        "--spectrum-output-dir", type=Path,
        default=project_root / "artifacts/spectrum_audited/FD001/stride50_seed42",
    )
    parser.add_argument(
        "--rebuilt-data-dir", type=Path,
        default=project_root / "artifacts/data_window40_stride50/split_seed42/FD001",
        help="verified new window data; images generated online from alignment spectrum weights",
    )
    parser.add_argument(
        "--alignment-checkpoint", type=Path,
        default=project_root / "artifacts/svlma_alignment/FD001/stride50_seed42/best.pt",
        help="trained DKE96+linear projector checkpoint (required by default)",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--cached-text-tokens", type=int, default=512)
    parser.add_argument("--projector-architecture", choices=["linear", "legacy_mlp"], default="linear")
    parser.add_argument("--train-stride", type=int, default=50)
    parser.add_argument("--validation-stride", type=int, default=50)
    parser.add_argument("--projector-epochs", type=int, default=5)
    parser.add_argument("--projector-learning-rate", type=float, default=1e-3)
    parser.add_argument("--projector-batch-size", type=int, default=128)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-projector-training", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
