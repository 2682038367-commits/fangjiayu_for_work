"""Supervised RUL comparison for random ViT and MAE-pretrained encoders."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import prepare_cmapss_data
from .mae import TemporalViTRegressor, VisionTransformerConfig
from .metrics import regression_metrics
from .spectrum_cache import CachedSpectrumDataset
from .training import clipped_metrics, seed_everything, write_prediction_svg, write_predictions
from .rul_protocol import TARGET_DIVISOR, TARGET_SCALE_STATUS


def predict(
    model: TemporalViTRegressor,
    loader: DataLoader,
    device: torch.device,
    target_divisor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = ([], [], [], [])
    with torch.inference_mode():
        for batch in loader:
            predictions = model(batch["x"].to(device), batch["image"].to(device)).cpu().numpy()
            output[0].append(predictions * target_divisor)
            output[1].append(batch["target"].numpy())
            output[2].append(batch["unit"].numpy())
            output[3].append(batch["cycle"].numpy())
    return tuple(np.concatenate(parts) for parts in output)  # type: ignore[return-value]


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = prepare_cmapss_data(args.data_dir, args.dataset, seed=args.split_seed)
    cached = {
        name: CachedSpectrumDataset(dataset, args.cache_dir / f"{name}.npy")
        for name, dataset in (("train", data.train), ("val", data.val), ("test", data.test))
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(cached["train"], args.batch_size, shuffle=True, generator=generator),
        "val": DataLoader(cached["val"], args.batch_size, shuffle=False),
        "test": DataLoader(cached["test"], args.batch_size, shuffle=False),
    }
    vision_config = VisionTransformerConfig()
    model = TemporalViTRegressor(vision_config=vision_config).to(device)
    temporal_checkpoint = args.temporal_checkpoint or Path("artifacts") / "temporal" / args.dataset / f"seed{args.seed}" / "best.pt"
    temporal_state = torch.load(temporal_checkpoint, map_location=device, weights_only=False)
    model.temporal.load_state_dict(temporal_state["model_state"])
    mae_checkpoint: Path | None = None
    if args.encoder == "mae":
        mae_checkpoint = args.mae_checkpoint or Path("artifacts") / "mae_pretrain" / args.dataset / f"seed{args.seed}" / "best.pt"
        mae_state = torch.load(mae_checkpoint, map_location=device, weights_only=False)
        model.visual.load_state_dict(mae_state["encoder_state"])

    optimizer = torch.optim.Adam(
        [
            {"params": model.temporal.parameters(), "lr": args.temporal_learning_rate},
            {"params": model.visual.parameters(), "lr": args.visual_learning_rate},
            {"params": model.fusion.parameters(), "lr": args.visual_learning_rate},
        ]
    )
    loss_function = nn.MSELoss()
    output_dir = args.output_dir or Path("artifacts") / f"temporal_{args.encoder}" / args.dataset / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = math.inf
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loaders["train"]:
            inputs = batch["x"].to(device)
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs, images)
            loss = loss_function(predictions, targets / TARGET_DIVISOR)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(targets)
            count += len(targets)
        val_predictions, val_targets, _, _ = predict(model, loaders["val"], device, TARGET_DIVISOR)
        val_metrics = clipped_metrics(val_targets, val_predictions, 125)
        train_rmse = math.sqrt(total / count) * TARGET_DIVISOR
        history.append({"epoch": epoch, "train_rmse": train_rmse, **{f"val_{key}": value for key, value in val_metrics.items()}})
        print(
            f"encoder={args.encoder} epoch={epoch:02d}/{args.epochs} "
            f"train_rmse={train_rmse:.4f} val_rmse={val_metrics['rmse']:.4f}",
            flush=True,
        )
        if val_metrics["rmse"] < best:
            best = val_metrics["rmse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_divisor": TARGET_DIVISOR,
                    "target_scale_status": TARGET_SCALE_STATUS,
                    "vision_encoder_state": model.visual.state_dict(),
                    "vision_config": vision_config.to_dict(),
                    "encoder": args.encoder,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions, targets, units, cycles = predict(model, loaders["test"], device, checkpoint["target_divisor"])
    result = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["val_metrics"],
        "test": clipped_metrics(targets, predictions, 125),
        "test_raw": regression_metrics(targets, predictions),
        "test_prediction_count": len(predictions),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "epochs": args.epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "temporal_learning_rate": args.temporal_learning_rate,
        "visual_learning_rate": args.visual_learning_rate,
        "cache_dir": str(args.cache_dir),
        "temporal_checkpoint": str(temporal_checkpoint),
        "mae_checkpoint": str(mae_checkpoint) if mae_checkpoint else None,
        "vision_config": vision_config.to_dict(),
    }
    write_predictions(output_dir / "test_predictions.csv", units, cycles, targets, predictions, 125)
    write_prediction_svg(output_dir / "test_predictions.svg", units, targets, predictions, 125)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=["vit", "mae"])
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/spectrum_cache/FD001/split_seed42"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temporal-checkpoint", type=Path)
    parser.add_argument("--mae-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temporal-learning-rate", type=float, default=0.0002)
    parser.add_argument("--visual-learning-rate", type=float, default=0.001)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
