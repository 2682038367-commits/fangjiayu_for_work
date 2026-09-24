#!/usr/bin/env python3
"""Evaluate the published LSTM on real C-MAPSS windows using saved splits."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
FIELDS = ("dataset", "evaluator_seed", "rmse", "mae", "rul_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["FD003", "FD004"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 13, 23, 33, 43])
    cli = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(UPSTREAM))
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from args import args as upstream_args
    from data.CMAPSSDataset import CMAPSSDataset
    from eva_regressor import predictive_score_metrics
    import wandb
    sys.argv = saved_argv

    result_path = ROOT / "results" / "real_data_baseline_w48.csv"
    rows = []
    if result_path.exists():
        with result_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    completed = {(row["dataset"], int(row["evaluator_seed"])) for row in rows}

    wandb.init(project="MetaIndux-TS-real-data-baseline", mode="disabled")
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        for dataset in cli.datasets:
            config_path = ROOT / "configs" / f"evaluation_{dataset.lower()}_w48_public_code.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            settings = config["evaluation"]
            with np.load(ROOT / config["outputs"]["split_indices"]) as split:
                train_idx = split["predictive_train"].copy()
                dev_idx = split["predictive_dev"].copy()

            cmaps = CMAPSSDataset(dataset, 48, [1000])
            real = {
                "data": cmaps.get_feature_slice(cmaps.get_train_data()),
                "label": cmaps.get_label_slice(cmaps.get_train_data()),
            }
            test_data, test_label = cmaps.get_last_data_slice(cmaps.get_test_data())
            if len(train_idx) + len(dev_idx) != len(real["data"]):
                raise ValueError(f"{dataset}: saved split does not match real data")

            upstream_args.dataset = dataset
            upstream_args.window_size = 48
            upstream_args.input_size = 14
            upstream_args.output_size = 1
            upstream_args.eva_epoch = int(settings["evaluator_epochs"])
            upstream_args.device = settings["device"]
            upstream_args.model_name = "DiffUnet_fre"
            checkpoint_dir = ROOT / "checkpoints" / "evaluators" / "real_data_baseline_w48"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            for seed in cli.seeds:
                if (dataset, seed) in completed:
                    print(f"skip completed {dataset} seed={seed}", flush=True)
                    continue
                print(f"evaluate real data {dataset} seed={seed}", flush=True)
                upstream_args.save_path = str(checkpoint_dir / f"{dataset}_eval{seed}.pth")
                rmse, mae, score = predictive_score_metrics(
                    upstream_args,
                    {"data": test_data, "label": test_label},
                    real,
                    eval_seed=seed,
                    split_indices=(train_idx, dev_idx),
                )
                rows.append({
                    "dataset": dataset,
                    "evaluator_seed": seed,
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "rul_score": float(score),
                })
                with result_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"{dataset} seed={seed} RMSE={rmse:.4f}", flush=True)
    finally:
        os.chdir(previous_cwd)
        wandb.finish()


if __name__ == "__main__":
    main()
