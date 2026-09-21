#!/usr/bin/env python3
"""Uniform batch32 + figure12_v1 input-level modality ablation across FD001-FD004.

Produces a strictly comparable Fig. 6 four-subset table. Every subset uses the same
optimization budget (batch_size=32, 30 epochs, Adam lr=0.002, no freeze), the same
figure12_v1 text protocol, the same input-level ablation definition, and the same
initial temporal weights for the three modality groups. Three seeds (42/52/62);
validation set is the only model-selection / aggregation criterion.

FD003 already has its batch32 figure12_v1 3-seed input ablation and is reused here.
FD001/FD002/FD004 are brought up to the same protocol: train a batch32 temporal
checkpoint, generate figure12_v1 text-only/visual-only caches, and train the three
modality groups across three seeds.

Idempotent: existing results are reused after a protocol check; missing stages are
run on GPU. Outputs land under artifacts/figure12_batch32_ablation/<dataset>/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Qwen weights are cached locally; force offline so transformers does not probe
# huggingface.co during tokenizer/model loading (is_base_mistral network check).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ts_mllm import training as temporal_training
from ts_mllm import qwen_cache
from ts_mllm import training_tmaf


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT.parent / "data" / "CMAPSSData"


def P(rel: str) -> Path:
    """Resolve a repo-relative path string (already prefixed with 'artifacts/')."""
    return ROOT / rel

SEEDS = (42, 52, 62)
SPLIT_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.002
PROMPT_PROFILE = "figure12_v1"

# Per-subset fixed main-experiment inputs (figure12_v1 alignment/cache already exist;
# the vision/MAE checkpoint and data are shared with the fixed main protocol).
SUBSETS = {
    "FD001": {
        "normalization": "global_minmax",
        "data_dir": "artifacts/data_window40_stride50/split_seed42/FD001",
        "vision_checkpoint": "artifacts/fd001_mae_stride50_repair/seed42/mae/best.pt",
        "figure12_alignment": "artifacts/figure12_v1/FD001/seed42/alignment/best.pt",
        "figure12_full_cache": "artifacts/figure12_v1/FD001/seed42/qwen",
    },
    "FD002": {
        "normalization": "condition_minmax",
        "data_dir": "artifacts/data_condition_minmax/split_seed42/FD002",
        "vision_checkpoint": "artifacts/mae_condition_minmax/FD002/stride50_seed42/best.pt",
        "figure12_alignment": "artifacts/figure12_v1/FD002/seed42/alignment/best.pt",
        "figure12_full_cache": "artifacts/figure12_v1/FD002/seed42/qwen",
    },
    "FD004": {
        "normalization": "condition_minmax",
        "data_dir": "artifacts/data_condition_minmax/split_seed42/FD004",
        "vision_checkpoint": "artifacts/mae_condition_minmax/FD004/stride50_seed42/best.pt",
        "figure12_alignment": "artifacts/figure12_v1/FD004/seed42/alignment/best.pt",
        "figure12_full_cache": "artifacts/figure12_v1/FD004/seed42/qwen",
    },
}

# FD003 already has a batch32 figure12_v1 3-seed input ablation at these locations.
FD003_RESULTS = {
    "full": {
        "cache": "artifacts/fd003_figure12_v1/seed42/qwen",
        "seed42": "artifacts/fd003_figure12_v1/seed42/tmaf_batch32",
        "seed52": "artifacts/fd003_input_ablation/seed52/tmaf_full",
        "seed62": "artifacts/fd003_input_ablation/seed62/tmaf_full",
    },
    "no_visual": {
        "cache": "artifacts/fd003_input_ablation/seed42/no_visual_qwen",
        "seed42": "artifacts/fd003_input_ablation/seed42/tmaf_no_visual",
        "seed52": "artifacts/fd003_input_ablation/seed52/tmaf_no_visual",
        "seed62": "artifacts/fd003_input_ablation/seed62/tmaf_no_visual",
    },
    "no_text": {
        "cache": "artifacts/fd003_input_ablation/seed42/no_text_qwen",
        "seed42": "artifacts/fd003_input_ablation/seed42/tmaf_no_text",
        "seed52": "artifacts/fd003_input_ablation/seed52/tmaf_no_text",
        "seed62": "artifacts/fd003_input_ablation/seed62/tmaf_no_text",
    },
}
MODALITY_ORDER = ("full", "no_visual", "no_text")
MODALITY_LABELS = {
    "full": "full (temporal + visual + textual)",
    "no_visual": "w/o Visual (temporal + textual)",
    "no_text": "w/o Textual (temporal + visual)",
}
CACHE_INPUT_MODALITY = {"full": "multimodal", "no_visual": "text_only", "no_text": "visual_only"}


def temporal_args(ds: str, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=ds,
        data_dir=DATA_DIR,
        output_dir=output_dir,
        device="auto",
        seed=SPLIT_SEED,
        split_seed=SPLIT_SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        train_stride=50,
        validation_stride=50,
        rebuilt_data_dir=P(SUBSETS[ds]["data_dir"]),
        validation_fraction=0.2,
        few_shot_fraction=1.0,
        num_workers=0,
        torch_threads=4,
    )


def cache_args(ds: str, input_modality: str, output_dir: Path) -> argparse.Namespace:
    spec = SUBSETS[ds]
    return argparse.Namespace(
        seed=SPLIT_SEED,
        split_seed=SPLIT_SEED,
        dataset=ds,
        data_dir=DATA_DIR,
        model_name="Qwen/Qwen3-0.6B",
        local_files_only=True,
        max_text_tokens=512,
        cached_text_tokens=512,
        projector_architecture="linear",
        train_stride=50,
        validation_stride=50,
        projector_epochs=5,
        projector_learning_rate=1e-3,
        projector_batch_size=128,
        inference_batch_size=16,
        workers=2,
        prompt_profile=PROMPT_PROFILE,
        input_modality=input_modality,
        output_dir=output_dir,
        spectrum_output_dir=None,
        spectrum_cache_dir=ARTIFACTS / "spectrum_cache" / ds / "split_seed42",
        rebuilt_data_dir=P(spec["data_dir"]),
        vision_checkpoint=P(spec["vision_checkpoint"]),
        alignment_checkpoint=P(spec["figure12_alignment"]),
        projector_checkpoint=None,
        skip_projector_training=False,
    )


def tmaf_args(ds: str, modality: str, cache_dir: Path, temporal_checkpoint: Path,
              seed: int, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=ds,
        data_dir=DATA_DIR,
        cache_dir=cache_dir,
        temporal_checkpoint=temporal_checkpoint,
        output_dir=output_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        temporal_learning_rate=None,
        freeze_temporal_epochs=0,
        rul_cap=125,
        seed=seed,
        split_seed=SPLIT_SEED,
        num_workers=2,
        rebuilt_data_dir=P(SUBSETS[ds]["data_dir"]),
        train_stride=50,
        validation_stride=50,
        context_mode="global_broadcast",
        global_pooling="mean",
        token_mode="full",
        shuffle_seed=2026,
        output_bias_init="default",
    )


def ensure_temporal(ds: str) -> Path:
    out = ARTIFACTS / "figure12_batch32_ablation" / ds / "temporal"
    best = out / "best.pt"
    if best.exists():
        print(f"[{ds}] reuse temporal {best}", flush=True)
        return best
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{ds}] train temporal batch32 ...", flush=True)
    temporal_training.train(temporal_args(ds, out))
    return best


def ensure_cache(ds: str, modality: str) -> Path:
    if modality == "full":
        return P(SUBSETS[ds]["figure12_full_cache"])
    out = ARTIFACTS / "figure12_batch32_ablation" / ds / f"qwen_{modality}"
    if (out / "manifest.json").exists():
        print(f"[{ds}] reuse cache {modality}: {out}", flush=True)
        return out
    print(f"[{ds}] generate cache {modality} ...", flush=True)
    qwen_cache.run(cache_args(ds, CACHE_INPUT_MODALITY[modality], out))
    return out


def ensure_tmaf(ds: str, modality: str, cache_dir: Path, temporal_checkpoint: Path,
                seed: int) -> dict:
    out = ARTIFACTS / "figure12_batch32_ablation" / ds / f"tmaf_{modality}" / f"seed{seed}"
    result_path = out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("seed") != seed or result.get("split_seed") != SPLIT_SEED \
                or result.get("batch_size") != BATCH_SIZE or result.get("token_mode") != "full":
            raise ValueError(f"incompatible existing result {result_path}")
        print(f"[{ds}] reuse tmaf {modality}/seed{seed}: {out}", flush=True)
        return result
    print(f"[{ds}] train tmaf {modality}/seed{seed} ...", flush=True)
    return training_tmaf.train(tmaf_args(ds, modality, cache_dir, temporal_checkpoint, seed, out))


def load_fd003_result(modality: str, seed: int) -> dict:
    path = P(FD003_RESULTS[modality][f"seed{seed}"])
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def load_result(ds: str, modality: str, seed: int) -> dict:
    if ds == "FD003":
        return load_fd003_result(modality, seed)
    out = ARTIFACTS / "figure12_batch32_ablation" / ds / f"tmaf_{modality}" / f"seed{seed}"
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def run_subset(ds: str) -> None:
    temporal_checkpoint = ensure_temporal(ds)
    for modality in MODALITY_ORDER:
        cache_dir = ensure_cache(ds, modality)
        for seed in SEEDS:
            ensure_tmaf(ds, modality, cache_dir, temporal_checkpoint, seed)


def aggregate() -> dict:
    runs = []
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        for modality in MODALITY_ORDER:
            for seed in SEEDS:
                result = load_result(ds, modality, seed)
                runs.append({
                    "dataset": ds, "modality": modality, "seed": seed,
                    "label": MODALITY_LABELS[modality],
                    "best_epoch": result["best_epoch"],
                    "validation": result["validation"],
                    "test": result["test"],
                })

    def stats(ds: str, modality: str, split: str, metric: str) -> dict:
        vals = [r[split][metric] for r in runs if r["dataset"] == ds and r["modality"] == modality]
        return {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    table = {}
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        table[ds] = {}
        for modality in MODALITY_ORDER:
            table[ds][modality] = {
                "validation": {m: stats(ds, modality, "validation", m) for m in ("rmse", "mae", "score")},
                "test": {m: stats(ds, modality, "test", m) for m in ("rmse", "mae", "score")},
            }

    summary = {
        "title": "FD001-FD004 input-level modality ablation, uniform batch32 + figure12_v1, 3 seeds",
        "protocol": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "temporal_learning_rate": LEARNING_RATE,
            "freeze_temporal_epochs": 0,
            "prompt_profile": PROMPT_PROFILE,
            "context_mode": "global_broadcast",
            "global_pooling": "mean",
            "train_stride": 50,
            "validation_stride": 50,
            "target_divisor": 125.0,
            "seeds": list(SEEDS),
            "split_seed": SPLIT_SEED,
            "std_convention": "sample standard deviation (ddof=1)",
            "selection_and_aggregation_criterion": "validation set only",
            "note": (
                "temporal branch shares the same initial checkpoint across seeds and "
                "modalities but is jointly fine-tuned (temporal_learning_rate=0.002, no freeze)"
            ),
        },
        "normalization": {ds: SUBSETS[ds]["normalization"] if ds in SUBSETS else "global_minmax"
                          for ds in ("FD001", "FD002", "FD003", "FD004")},
        "runs": runs,
        "table": table,
    }
    out = ARTIFACTS / "FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def print_table(summary: dict) -> None:
    print("\n==== Validation RMSE (mean +- sample std over 3 seeds) ====")
    header = "| subset | full | w/o Visual | w/o Textual |"
    print(header)
    print("|---|---|---|---|")
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        cells = []
        for modality in MODALITY_ORDER:
            s = summary["table"][ds][modality]["validation"]["rmse"]
            cells.append(f"{s['mean']:.3f} +- {s['std']:.3f}")
        print(f"| {ds} | {' | '.join(cells)} |")
    print("\n==== Test RMSE (descriptive only, not for model selection) ====")
    print(header)
    print("|---|---|---|---|")
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        cells = []
        for modality in MODALITY_ORDER:
            s = summary["table"][ds][modality]["test"]["rmse"]
            cells.append(f"{s['mean']:.3f} +- {s['std']:.3f}")
        print(f"| {ds} | {' | '.join(cells)} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", choices=["FD001", "FD002", "FD004"],
                        default=["FD001", "FD002", "FD004"])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only:
        for ds in args.subsets:
            run_subset(ds)
    summary = aggregate()
    print_table(summary)
    print(f"\nwrote {ARTIFACTS / 'FIGURE6_FOUR_SUBSET_BATCH32_ABLATION.json'}", flush=True)


if __name__ == "__main__":
    main()
