"""GPU training entry point for TMAF over cached Qwen features."""

from __future__ import annotations

import argparse
import json
import shutil
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import prepare_cmapss_data
from .metrics import regression_metrics
from .qwen_dataset import CachedQwenDataset
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from .training import clipped_metrics, seed_everything, write_prediction_svg, write_predictions
from .rul_protocol import TARGET_DIVISOR, TARGET_SCALE_STATUS
from .bias_initialization import initialize_output_bias


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("TMAF training requires CUDA; CPU fallback is disabled")
    return torch.device("cuda")


def set_temporal_phase(
    model: TemporalMultimodalAttentionFusion, epoch: int, freeze_epochs: int
) -> bool:
    frozen = epoch <= freeze_epochs
    model.temporal.requires_grad_(not frozen)
    # This head belongs to the loaded unimodal checkpoint, but TMAF uses its
    # own regression head. Never optimize the unused unimodal head.
    model.temporal.regression_head.requires_grad_(False)
    if frozen:
        # Disable temporal dropout too: frozen features must be stable while
        # the newly initialized attention/fusion/head warm up.
        model.temporal.eval()
    return frozen


@torch.inference_mode()
def predict(
    model: TemporalMultimodalAttentionFusion,
    loader: DataLoader,
    device: torch.device,
    target_divisor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    collected: tuple[list[np.ndarray], ...] = ([], [], [], [])
    for batch in loader:
        output = model(
            batch["x"].to(device, non_blocking=True),
            batch["llm_tokens"].to(device, non_blocking=True),
            batch["llm_mask"].to(device, non_blocking=True),
        ).cpu().numpy()
        collected[0].append(output * target_divisor)
        collected[1].append(batch["target"].numpy())
        collected[2].append(batch["unit"].numpy())
        collected[3].append(batch["cycle"].numpy())
    return tuple(np.concatenate(parts) for parts in collected)  # type: ignore[return-value]


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    device = require_cuda()
    bundle = prepare_cmapss_data(
        args.data_dir, args.dataset, seed=args.split_seed,
        train_stride=args.train_stride, validation_stride=args.validation_stride,
    )
    base_datasets = {name: getattr(bundle, name) for name in ("train", "val", "test")}
    data_source_hash = None
    if args.rebuilt_data_dir is not None:
        source_manifest = json.loads((args.rebuilt_data_dir / "manifest.json").read_text())
        if (source_manifest["dataset"] != args.dataset or source_manifest["split_seed"] != args.split_seed
                or source_manifest["train_sample_stride"] != args.train_stride
                or source_manifest["validation_sample_stride"] != args.validation_stride
                or source_manifest["rul_cap"] != args.rul_cap):
            raise ValueError("TMAF arguments do not match exported data protocol")
        data_source_hash = file_sha256(args.rebuilt_data_dir / "manifest.json")
        base_datasets = {name: RebuiltWindowDataset(args.rebuilt_data_dir, name) for name in base_datasets}
        cache_manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        if cache_manifest["data_manifest_sha256"] != data_source_hash:
            raise ValueError("TMAF data source does not match audited Qwen cache")
        if cache_manifest["cache_config"]["max_text_tokens"] != 512 or cache_manifest["text_dim"] != 96:
            raise ValueError("audited TMAF requires full512 text input and DKE96")
        if cache_manifest["cache_config"]["cached_text_tokens"] != 512:
            raise ValueError("audited TMAF requires all text output tokens")
        for filename, expected_hash in cache_manifest["cache_file_sha256"].items():
            if file_sha256(args.cache_dir / filename) != expected_hash:
                raise ValueError(f"Qwen cache checksum mismatch: {filename}")
    datasets = {
        name: CachedQwenDataset(
            base,
            args.cache_dir,
            name,
            token_mode=args.token_mode,
            shuffle_seed=args.shuffle_seed,
        )
        for name, base in base_datasets.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        for name, dataset in datasets.items()
    }

    config = TMAFConfig(context_mode=args.context_mode, global_pooling=getattr(args, "global_pooling", "mean"))
    model = TemporalMultimodalAttentionFusion(config).to(device)
    temporal_checkpoint = torch.load(
        args.temporal_checkpoint, map_location="cpu", weights_only=False
    )
    if data_source_hash is not None and temporal_checkpoint.get("data_manifest_sha256") != data_source_hash:
        raise ValueError("temporal checkpoint was not trained on the audited data source")
    if data_source_hash is not None and temporal_checkpoint.get("target_divisor", 1.0) != TARGET_DIVISOR:
        raise ValueError("audited scale-B TMAF requires a scale-B temporal checkpoint")
    model.load_temporal_checkpoint(temporal_checkpoint["model_state"])
    training_mean_cycles = float(np.mean([float(base_datasets["train"][i]["target"]) for i in range(len(base_datasets["train"]))]))
    initialization = initialize_output_bias(model, getattr(args, "output_bias_init", "default"), training_mean_cycles, TARGET_DIVISOR)
    temporal_lr = args.temporal_learning_rate or args.learning_rate
    temporal_parameters = list(model.temporal.parameters())
    temporal_ids = {id(parameter) for parameter in temporal_parameters}
    fusion_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in temporal_ids
    ]
    optimizer = torch.optim.Adam([
        {"params": temporal_parameters, "lr": temporal_lr, "name": "temporal"},
        {"params": fusion_parameters, "lr": args.learning_rate, "name": "fusion"},
    ])
    loss_function = nn.MSELoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    if data_source_hash is not None and checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite audited model: {args.output_dir}")
    (args.output_dir / "initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    history: list[dict[str, float | int]] = []
    best_val_rmse = math.inf
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        temporal_frozen = set_temporal_phase(model, epoch, args.freeze_temporal_epochs)
        squared_error = 0.0
        count = 0
        for batch in loaders["train"]:
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                batch["x"].to(device, non_blocking=True),
                batch["llm_tokens"].to(device, non_blocking=True),
                batch["llm_mask"].to(device, non_blocking=True),
            )
            loss = loss_function(predictions, targets / TARGET_DIVISOR)
            loss.backward()
            optimizer.step()
            squared_error += float(loss.detach()) * len(targets)
            count += len(targets)
        val_predictions, val_targets, _, _ = predict(model, loaders["val"], device, TARGET_DIVISOR)
        val_metrics = clipped_metrics(val_targets, val_predictions, args.rul_cap)
        train_rmse = math.sqrt(squared_error / count) * TARGET_DIVISOR
        record = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "temporal_frozen": temporal_frozen,
            "temporal_lr": 0.0 if temporal_frozen else temporal_lr,
            "fusion_lr": args.learning_rate,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        print(
            f"epoch={epoch:02d}/{args.epochs} train_rmse={train_rmse:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f}",
            flush=True,
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "initialization": initialization,
                    "target_divisor": TARGET_DIVISOR,
                    "target_scale_status": TARGET_SCALE_STATUS,
                    "data_manifest_sha256": data_source_hash,
                    "model_config": config.to_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "temporal_checkpoint": str(args.temporal_checkpoint),
                    "temporal_checkpoint_sha256": file_sha256(args.temporal_checkpoint),
                    "qwen_cache": str(args.cache_dir),
                    "train_stride": args.train_stride,
                    "validation_stride": args.validation_stride,
                    "fidelity_status": "assumption_based_not_verified_paper_reproduction",
                    "temporal_learning_rate": temporal_lr,
                    "fusion_learning_rate": args.learning_rate,
                    "freeze_temporal_epochs": args.freeze_temporal_epochs,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions, targets, units, cycles = predict(model, loaders["test"], device, checkpoint["target_divisor"])
    result: dict[str, object] = {
        "dataset": args.dataset,
        "prompt_profile": cache_manifest.get("prompt_profile", "legacy") if data_source_hash else "legacy",
        "rebuilt_data_dir": str(args.rebuilt_data_dir.resolve()) if args.rebuilt_data_dir else None,
        "initialization": initialization,
        "target_divisor": checkpoint["target_divisor"],
        "target_scale_status": TARGET_SCALE_STATUS,
        "data_manifest_sha256": data_source_hash,
        "qwen_cache_manifest_sha256": file_sha256(args.cache_dir / "manifest.json") if data_source_hash else None,
        "token_mode": args.token_mode,
        "train_stride": args.train_stride,
        "validation_stride": args.validation_stride,
        "fidelity_status": "assumption_based_not_verified_paper_reproduction",
        "shuffle_seed": args.shuffle_seed if args.token_mode == "shuffled" else None,
        "device": torch.cuda.get_device_name(0),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["val_metrics"],
        "test": clipped_metrics(targets, predictions, args.rul_cap),
        "test_raw": regression_metrics(targets, predictions),
        "test_prediction_count": int(len(predictions)),
        "prediction_finite": bool(np.isfinite(predictions).all()),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temporal_learning_rate": temporal_lr,
        "freeze_temporal_epochs": args.freeze_temporal_epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "model_config": config.to_dict(),
        "temporal_checkpoint": str(args.temporal_checkpoint),
        "temporal_checkpoint_sha256": file_sha256(args.temporal_checkpoint),
        "qwen_cache": str(args.cache_dir),
    }
    write_predictions(
        args.output_dir / "test_predictions.csv", units, cycles, targets, predictions, args.rul_cap
    )
    write_prediction_svg(
        args.output_dir / "test_predictions.svg", units, targets, predictions, args.rul_cap
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if data_source_hash:
        scaler_path = args.rebuilt_data_dir / "scaler.json"
        if file_sha256(scaler_path) != source_manifest["scaler_sha256"]:
            raise ValueError("exported scaler checksum mismatch")
        shutil.copyfile(scaler_path, args.output_dir / "scaler.json")
    else:
        bundle.scaler.save(args.output_dir / "scaler.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train the TS-MLLM TMAF regression model")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=project_root / "artifacts/qwen_multimodal/FD001/seed42"
    )
    parser.add_argument(
        "--temporal-checkpoint", type=Path,
        default=project_root / "artifacts/temporal/FD001/seed42/best.pt"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="defaults to a separate directory for each token mode and seed",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--temporal-learning-rate", type=float)
    parser.add_argument("--freeze-temporal-epochs", type=int, default=0)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rebuilt-data-dir", type=Path)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=1)
    parser.add_argument("--context-mode", choices=["tokens", "global_broadcast"], default="tokens")
    parser.add_argument("--global-pooling", choices=["mean", "last_text"], default="mean",
                        help="Paper-unspecified global vector extraction; last_text excludes visual prefix")
    parser.add_argument(
        "--token-mode",
        choices=["full", "visual_only", "text_only", "shuffled"],
        default="full",
        help="Qwen-token modality ablation; shuffled permutes tokens across samples",
    )
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--output-bias-init", choices=["default", "train_mean"], default="default",
                        help="paper-unspecified final regression bias initialization experiment")
    args = parser.parse_args()
    if args.global_pooling == "last_text" and (args.context_mode != "global_broadcast" or args.token_mode == "visual_only"):
        parser.error("last_text requires global_broadcast and valid text tokens")
    if not 0 <= args.freeze_temporal_epochs < args.epochs:
        parser.error("freeze-temporal-epochs must be >= 0 and less than epochs")
    if args.temporal_learning_rate is not None and args.temporal_learning_rate <= 0:
        parser.error("temporal-learning-rate must be positive")
    if args.output_dir is None:
        root = project_root / "artifacts"
        if args.freeze_temporal_epochs or args.temporal_learning_rate is not None:
            root = root / "tmaf_staged" / args.token_mode
        elif args.context_mode != "tokens" or args.train_stride != 1 or args.validation_stride != 1:
            root = root / "tmaf_audited" / args.context_mode / f"stride{args.train_stride}_val{args.validation_stride}" / args.token_mode
        elif args.token_mode == "full":
            root = root / "tmaf"
        else:
            root = root / "tmaf_ablation" / args.token_mode
        if args.global_pooling != "mean":
            root = root / args.global_pooling
        args.output_dir = root / args.dataset / f"seed{args.seed}"
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
