#!/usr/bin/env python3
"""Evaluate existing FD002 synthetic datasets with a controlled seed design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "upstream"
RAW_FIELDS = [
    "generation_seed",
    "evaluator_seed",
    "rmse",
    "mae",
    "rul_score",
    "discriminative_score",
    "fake_accuracy",
    "real_accuracy",
    "predictive_seconds",
    "discriminative_seconds",
]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_hash(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array, dtype=np.int64).tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evaluation_fd002_stability.yaml",
    )
    cli = parser.parse_args()
    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(UPSTREAM))

    # upstream/args.py parses argv during import.
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from args import args as upstream_args
    from data.CMAPSSDataset import CMAPSSDataset
    from eva_regressor import predictive_score_metrics
    from measure_score.Utils.discriminative_metric import discriminative_score_metrics
    import wandb
    sys.argv = saved_argv

    dataset = experiment["dataset"]
    window_size = int(experiment["window_size"])
    generation_seeds = list(map(int, experiment["generation_seeds"]))
    evaluator_seeds = list(map(int, experiment["evaluator_seeds"]))
    fixed_split_seed = int(experiment["fixed_split_seed"])

    previous_cwd = Path.cwd()
    os.chdir(UPSTREAM)
    try:
        data = CMAPSSDataset(dataset, window_size, [1000])
        train_data = data.get_feature_slice(data.get_train_data())
        test_data, test_label = data.get_last_data_slice(data.get_test_data())
    finally:
        os.chdir(previous_cwd)

    sample_count = len(train_data)
    predictive_train_rate = float(evaluation.get("predictive_train_rate", 0.8))
    discriminator_train_rate = float(evaluation.get("discriminator_train_rate", 0.8))
    predictive_split_at = int(sample_count * predictive_train_rate)
    discriminator_split_at = int(sample_count * discriminator_train_rate)
    predictive_rng = np.random.RandomState(fixed_split_seed)
    predictive_perm = predictive_rng.permutation(sample_count)
    pred_train_idx = predictive_perm[:predictive_split_at]
    pred_dev_idx = predictive_perm[predictive_split_at:]

    discriminator_rng = np.random.RandomState(fixed_split_seed + 1)
    real_perm = discriminator_rng.permutation(sample_count)
    generated_perm = discriminator_rng.permutation(sample_count)
    real_train_idx = real_perm[:discriminator_split_at]
    real_test_idx = real_perm[discriminator_split_at:]
    gen_train_idx = generated_perm[:discriminator_split_at]
    gen_test_idx = generated_perm[discriminator_split_at:]

    split_path = project_path(outputs["split_indices"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_path,
        predictive_train=pred_train_idx,
        predictive_dev=pred_dev_idx,
        discriminator_real_train=real_train_idx,
        discriminator_real_test=real_test_idx,
        discriminator_generated_train=gen_train_idx,
        discriminator_generated_test=gen_test_idx,
    )
    split_arrays = [
        pred_train_idx, pred_dev_idx, real_train_idx, real_test_idx,
        gen_train_idx, gen_test_idx,
    ]

    raw_path = project_path(outputs["raw_csv"])
    existing: list[dict] = []
    if raw_path.exists() and evaluation.get("resume", True):
        with raw_path.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    completed = {
        (int(row["generation_seed"]), int(row["evaluator_seed"]))
        for row in existing
    }
    rows: list[dict] = existing

    upstream_args.dataset = dataset
    upstream_args.window_size = window_size
    upstream_args.input_size = 14
    upstream_args.output_size = 1
    upstream_args.eva_epoch = int(evaluation["evaluator_epochs"])
    upstream_args.device = evaluation["device"]
    upstream_args.model_name = "DiffUnet_fre"
    checkpoint_dir = project_path(outputs["evaluator_checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(project="MetaIndux-TS-evaluation-stability", mode="disabled")

    original_test = {"data": test_data, "label": test_label}
    real_numpy = train_data.numpy()
    os.chdir(UPSTREAM)
    try:
        for generation_seed in generation_seeds:
            synth_pattern = experiment.get(
                "synthetic_pattern",
                "outputs/FD002_w48_seed{generation_seed}.npz",
            )
            synth_path = project_path(
                synth_pattern.format(generation_seed=generation_seed)
            )
            with np.load(synth_path) as archive:
                generated = {
                    "data": archive["data"].copy(),
                    "label": archive["label"].copy(),
                }
            for evaluator_seed in evaluator_seeds:
                key = (generation_seed, evaluator_seed)
                if key in completed:
                    print(f"skip completed generation={generation_seed} evaluator={evaluator_seed}", flush=True)
                    continue
                print(f"evaluate generation={generation_seed} evaluator={evaluator_seed}", flush=True)
                upstream_args.save_path = str(
                    checkpoint_dir / f"gen{generation_seed}_eval{evaluator_seed}.pth"
                )

                started = time.time()
                rmse, mae, rul_score = predictive_score_metrics(
                    upstream_args,
                    original_test,
                    generated,
                    eval_seed=evaluator_seed,
                    split_indices=(pred_train_idx, pred_dev_idx),
                )
                predictive_seconds = time.time() - started

                started = time.time()
                discriminative, fake_accuracy, real_accuracy = discriminative_score_metrics(
                    real_numpy,
                    generated["data"],
                    eval_seed=evaluator_seed,
                    split_indices=(real_train_idx, real_test_idx, gen_train_idx, gen_test_idx),
                )
                discriminative_seconds = time.time() - started

                rows.append({
                    "generation_seed": generation_seed,
                    "evaluator_seed": evaluator_seed,
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "rul_score": float(rul_score),
                    "discriminative_score": float(discriminative),
                    "fake_accuracy": float(fake_accuracy),
                    "real_accuracy": float(real_accuracy),
                    "predictive_seconds": predictive_seconds,
                    "discriminative_seconds": discriminative_seconds,
                })
                write_csv(raw_path, rows, RAW_FIELDS)
    finally:
        os.chdir(previous_cwd)
        wandb.finish()

    numeric_rows = [
        {key: float(value) if key not in ("generation_seed", "evaluator_seed") else int(value)
         for key, value in row.items()}
        for row in rows
    ]
    summary_fields = [
        "metric", "generation_seed", "mean", "sample_std",
        "pooled_within_std", "between_generation_std_of_means",
        "corrected_generation_component_std", "total_std",
    ]
    summary_rows: list[dict] = []
    for metric in ("rmse", "discriminative_score"):
        means = []
        variances = []
        all_values = []
        for generation_seed in generation_seeds:
            values = np.array([
                row[metric] for row in numeric_rows
                if row["generation_seed"] == generation_seed
            ])
            if len(values) != len(evaluator_seeds):
                raise RuntimeError(f"incomplete evaluations for generation seed {generation_seed}")
            means.append(values.mean())
            variances.append(values.var(ddof=1))
            all_values.extend(values)
            summary_rows.append({
                "metric": metric,
                "generation_seed": generation_seed,
                "mean": values.mean(),
                "sample_std": values.std(ddof=1),
                "pooled_within_std": "",
                "between_generation_std_of_means": "",
                "corrected_generation_component_std": "",
                "total_std": "",
            })
        pooled_within_var = float(np.mean(variances))
        between_var = float(np.var(means, ddof=1))
        corrected_generation_var = max(0.0, between_var - pooled_within_var / len(evaluator_seeds))
        summary_rows.append({
            "metric": metric,
            "generation_seed": "aggregate",
            "mean": np.mean(means),
            "sample_std": "",
            "pooled_within_std": np.sqrt(pooled_within_var),
            "between_generation_std_of_means": np.sqrt(between_var),
            "corrected_generation_component_std": np.sqrt(corrected_generation_var),
            "total_std": np.std(all_values, ddof=1),
        })
    write_csv(project_path(outputs["summary_csv"]), summary_rows, summary_fields)

    manifest = {
        "status": "complete",
        "config": config,
        "sample_count": sample_count,
        "predictive_train_count": len(pred_train_idx),
        "predictive_dev_count": len(pred_dev_idx),
        "discriminator_train_count_per_class": len(real_train_idx),
        "discriminator_test_count_per_class": len(real_test_idx),
        "split_sha256": split_hash(split_arrays),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "evaluation_count": len(rows),
    }
    manifest_path = project_path(outputs["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"complete: {len(rows)} evaluations", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
