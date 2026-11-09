#!/usr/bin/env python3
"""Minimal repeated verification of the FD003 input-level modality ablation.

Fixes the FD003 data split and initial temporal checkpoint, and re-runs the three
input-level modality groups (full / w/o Visual / w/o Textual) under three random
seeds (42, 52, 62). All groups share the same initial temporal checkpoint, but the
temporal branch is jointly fine-tuned during training (temporal_learning_rate=0.002,
no freeze), so the seed variance reflects TMAF, regression-head, and temporal-branch
training trajectories. The data split, initial temporal checkpoint, prompt profile,
and frozen Qwen caches are held fixed; the modality comparison remains fair because
every group starts from the same initial weights. This is a repeated-measurement
check of the Fig. 6 modality contribution, not a change to TMAF, prompt, or data
processing.

Seed 42 reuses the existing audited input-ablation results; seeds 52/62 are
trained here into isolated directories under artifacts/fd003_input_ablation/.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ts_mllm.training_tmaf import parse_args, train


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

SEEDS = (42, 52, 62)
SPLIT_SEED = 42

DATA_MANIFEST_SHA = "b0cddcfd93fd487a69c996748629be9a23632dd021effd5dcd702232dcf32fb6"
TEMPORAL_SHA = "a1002727cefd7c8d1e5f3a148bd5a49a71870e7c1f2e82dc61b56af6ecc35708"

REBUILT_DATA = ARTIFACTS / "data_window40_stride50" / "split_seed42" / "FD003"
TEMPORAL_CKPT = ARTIFACTS / "fd003_batch_diagnostic" / "seed42" / "batch32" / "best.pt"

MODALITIES: dict[str, dict[str, object]] = {
    "full": {
        "label": "full (temporal + visual + textual)",
        "input_modality": "multimodal",
        "cache": ARTIFACTS / "fd003_figure12_v1" / "seed42" / "qwen",
        "seed42_output": ARTIFACTS / "fd003_figure12_v1" / "seed42" / "tmaf_batch32",
    },
    "no_visual": {
        "label": "w/o Visual (temporal + textual)",
        "input_modality": "text_only",
        "cache": ARTIFACTS / "fd003_input_ablation" / "seed42" / "no_visual_qwen",
        "seed42_output": ARTIFACTS / "fd003_input_ablation" / "seed42" / "tmaf_no_visual",
    },
    "no_text": {
        "label": "w/o Textual (temporal + visual)",
        "input_modality": "visual_only",
        "cache": ARTIFACTS / "fd003_input_ablation" / "seed42" / "no_text_qwen",
        "seed42_output": ARTIFACTS / "fd003_input_ablation" / "seed42" / "tmaf_no_text",
    },
}


def output_dir_for(seed: int, modality: str) -> Path:
    if seed == SPLIT_SEED:
        return Path(MODALITIES[modality]["seed42_output"])  # type: ignore[arg-type]
    return ARTIFACTS / "fd003_input_ablation" / f"seed{seed}" / f"tmaf_{modality}"


def build_args(seed: int, modality: str, output_dir: Path):
    args = copy.copy(BASE_ARGS)
    args.dataset = "FD003"
    args.cache_dir = Path(MODALITIES[modality]["cache"])  # type: ignore[arg-type]
    args.temporal_checkpoint = TEMPORAL_CKPT
    args.rebuilt_data_dir = REBUILT_DATA
    args.train_stride = 50
    args.validation_stride = 50
    args.context_mode = "global_broadcast"
    args.global_pooling = "mean"
    args.batch_size = 32
    args.seed = seed
    args.split_seed = SPLIT_SEED
    args.token_mode = "full"
    args.output_dir = output_dir
    return args


def validate_existing(result: dict, seed: int, modality: str, cache: Path) -> None:
    model_config = result.get("model_config", {})
    checks = {
        "dataset": result.get("dataset") == "FD003",
        "seed": result.get("seed") == seed,
        "split_seed": result.get("split_seed") == SPLIT_SEED,
        "batch_size": result.get("batch_size") == 32,
        "train_stride": result.get("train_stride") == 50,
        "validation_stride": result.get("validation_stride") == 50,
        "token_mode": result.get("token_mode") == "full",
        "context_mode": model_config.get("context_mode") == "global_broadcast",
        "global_pooling": model_config.get("global_pooling") == "mean",
        "data_manifest_sha256": result.get("data_manifest_sha256") == DATA_MANIFEST_SHA,
        "temporal_checkpoint_sha256": result.get("temporal_checkpoint_sha256") == TEMPORAL_SHA,
        "qwen_cache": Path(result.get("qwen_cache", "")).resolve() == cache.resolve(),
    }
    if not all(checks.values()):
        bad = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"incompatible existing result {modality}/seed{seed}: {bad}")


def main() -> None:
    runs: list[dict[str, object]] = []
    for seed in SEEDS:
        for modality, spec in MODALITIES.items():
            output_dir = output_dir_for(seed, modality)
            cache = Path(spec["cache"])  # type: ignore[arg-type]
            result_path = output_dir / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                validate_existing(result, seed, modality, cache)
                print(f"reuse {modality}/seed{seed}: {result_path}", flush=True)
            else:
                print(f"\n=== train {modality}/seed{seed} ===", flush=True)
                args = build_args(seed, modality, output_dir)
                result = train(args)
            runs.append(
                {
                    "seed": seed,
                    "modality": modality,
                    "input_modality": spec["input_modality"],
                    "label": spec["label"],
                    "output_dir": str(output_dir),
                    "qwen_cache": str(cache),
                    "best_epoch": result["best_epoch"],
                    "validation": result["validation"],
                    "test": result["test"],
                }
            )

    def aggregate(key: str, metric: str) -> dict[str, float]:
        values = {
            modality: [r[key][metric] for r in runs if r["modality"] == modality]
            for modality in MODALITIES
        }
        out: dict[str, float] = {}
        for modality, arr in values.items():
            out[f"{modality}_mean"] = float(np.mean(arr))
            out[f"{modality}_std"] = float(np.std(arr, ddof=1))
            out[f"{modality}_min"] = float(np.min(arr))
            out[f"{modality}_max"] = float(np.max(arr))
        return out

    summary = {
        "title": "FD003 input-level modality ablation, 3-seed repeated verification",
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "std_convention": "sample standard deviation (ddof=1)",
        "fixed": {
            "data_manifest_sha256": DATA_MANIFEST_SHA,
            "rebuilt_data_dir": str(REBUILT_DATA),
            "initial_temporal_checkpoint": str(TEMPORAL_CKPT),
            "initial_temporal_checkpoint_sha256": TEMPORAL_SHA,
            "temporal_learning_rate": BASE_ARGS.learning_rate,
            "freeze_temporal_epochs": BASE_ARGS.freeze_temporal_epochs,
            "temporal_note": (
                "temporal branch shares the same initial checkpoint across seeds and "
                "modalities but is jointly fine-tuned (temporal_learning_rate=0.002, no freeze)"
            ),
            "batch_size": 32,
            "epochs": BASE_ARGS.epochs,
            "learning_rate": BASE_ARGS.learning_rate,
            "context_mode": "global_broadcast",
            "global_pooling": "mean",
            "prompt_profile": "figure12_v1",
            "token_mode": "full (input modality encoded in cache)",
        },
        "runs": runs,
        "validation_aggregate": {
            "rmse": aggregate("validation", "rmse"),
            "mae": aggregate("validation", "mae"),
            "score": aggregate("validation", "score"),
        },
        "test_aggregate": {
            "rmse": aggregate("test", "rmse"),
            "mae": aggregate("test", "mae"),
            "score": aggregate("test", "score"),
        },
    }
    summary_path = ARTIFACTS / "FD003_INPUT_MODALITY_ABLATION_3SEED.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nwrote {summary_path}", flush=True)


BASE_ARGS = parse_args()


if __name__ == "__main__":
    main()
