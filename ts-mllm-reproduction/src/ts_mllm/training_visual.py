"""Train the temporal + RP/STFT/CWT visual baseline."""

from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import prepare_cmapss_data
from .model import PatchTransformerConfig
from .training import (
    TrainConfig,
    clipped_metrics,
    make_loader,
    predict,
    seed_everything,
    write_prediction_svg,
    write_predictions,
)
from .vision import TemporalVisualRegressor


from .rul_protocol import TARGET_DIVISOR, TARGET_SCALE_STATUS


def write_png(path: Path, image: torch.Tensor) -> None:
    """Save a [3,H,W] tensor as a dependency-free RGB PNG preview."""
    pixels = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    height, width, _ = pixels.shape
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body))

    scanlines = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def train(args: argparse.Namespace) -> dict[str, object]:
    config = TrainConfig(
        dataset=args.dataset,
        seed=args.seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_stride=args.train_stride,
        validation_fraction=args.validation_fraction,
        few_shot_fraction=args.few_shot_fraction,
        num_workers=args.num_workers,
        torch_threads=args.torch_threads,
    )
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    temporal_config = PatchTransformerConfig()
    bundle = prepare_cmapss_data(
        args.data_dir,
        config.dataset,
        seed=config.split_seed,
        validation_fraction=config.validation_fraction,
        few_shot_fraction=config.few_shot_fraction,
        window_size=temporal_config.window_size,
        train_stride=config.train_stride,
        rul_cap=config.rul_cap,
    )
    train_loader = make_loader(bundle.train, config.batch_size, True, config.seed, config.num_workers)
    val_loader = make_loader(bundle.val, config.batch_size, False, config.seed, config.num_workers)
    test_loader = make_loader(bundle.test, config.batch_size, False, config.seed, config.num_workers)

    model = TemporalVisualRegressor(temporal_config).to(device)
    temporal_checkpoint = args.temporal_checkpoint or (
        Path("artifacts") / "temporal" / config.dataset / f"seed{config.seed}" / "best.pt"
    )
    if temporal_checkpoint is not None:
        checkpoint = torch.load(temporal_checkpoint, map_location=device, weights_only=False)
        model.temporal.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    output_dir = args.output_dir or Path("artifacts") / "temporal_visual" / config.dataset / f"seed{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_val_rmse = math.inf
    started = time.time()

    # Store a visual sanity-check before optimization. Channels map to R/G/B as
    # RP/STFT/CWT and are directly inspectable with most image viewers.
    with torch.inference_mode():
        example = bundle.train[0]["x"].unsqueeze(0).to(device)
        write_png(output_dir / "spectrum_example_initial.png", model.spectrum(example)[0])

    for epoch in range(1, config.epochs + 1):
        model.train()
        squared_error_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs = batch["x"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_function(predictions, targets / TARGET_DIVISOR)
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach()) * len(targets)
            sample_count += len(targets)

        val_predictions, val_targets, _, _ = predict(model, val_loader, device, TARGET_DIVISOR)
        val_metrics = clipped_metrics(val_targets, val_predictions, config.rul_cap)
        train_rmse = math.sqrt(squared_error_sum / sample_count) * TARGET_DIVISOR
        record = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        print(
            f"epoch={epoch:02d}/{config.epochs} train_rmse={train_rmse:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f}",
            flush=True,
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_divisor": TARGET_DIVISOR,
                    "target_scale_status": TARGET_SCALE_STATUS,
                    "temporal_config": temporal_config.to_dict(),
                    "train_config": config.__dict__,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "visual_config": {
                        "image_size": 114,
                        "visual_dim": 128,
                        "encoder": "lightweight_cnn",
                        "channels": ["RP", "STFT", "Morlet-CWT"],
                        "stft_n_fft": 16,
                        "stft_hop_length": 4,
                        "cwt_scales": [1, 2, 3, 4, 6, 8, 12, 16],
                    },
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions, test_targets, test_units, test_cycles = predict(model, test_loader, device, checkpoint["target_divisor"])
    test_metrics = clipped_metrics(test_targets, test_predictions, config.rul_cap)
    from .metrics import regression_metrics

    raw_metrics = regression_metrics(test_targets, test_predictions)
    result: dict[str, object] = {
        "dataset": config.dataset,
        "model": "temporal_visual_cnn",
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["val_metrics"],
        "test": test_metrics,
        "test_raw": raw_metrics,
        "test_prediction_count": int(len(test_predictions)),
        "elapsed_seconds": time.time() - started,
        "train_config": config.__dict__,
        "temporal_config": temporal_config.to_dict(),
        "visual_config": checkpoint["visual_config"],
        "temporal_checkpoint": str(temporal_checkpoint) if temporal_checkpoint else None,
    }
    write_predictions(
        output_dir / "test_predictions.csv",
        test_units,
        test_cycles,
        test_targets,
        test_predictions,
        config.rul_cap,
    )
    write_prediction_svg(
        output_dir / "test_predictions.svg",
        test_units,
        test_targets,
        test_predictions,
        config.rul_cap,
    )
    with torch.inference_mode():
        example = bundle.test[0]["x"].unsqueeze(0).to(device)
        write_png(output_dir / "spectrum_example_trained.png", model.spectrum(example)[0])
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    bundle.scaler.save(output_dir / "scaler.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--few-shot-fraction", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument(
        "--temporal-checkpoint",
        type=Path,
        help="defaults to artifacts/temporal/<dataset>/seed<seed>/best.pt",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
