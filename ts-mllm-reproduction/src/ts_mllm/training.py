"""Training entry point for the pure temporal TS-MLLM baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CMapssWindowDataset, prepare_cmapss_data
from .metrics import regression_metrics
from .model import PatchTransformer, PatchTransformerConfig
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .rul_protocol import TARGET_DIVISOR, TARGET_SCALE_STATUS
from .bias_initialization import state_digest


@dataclass(frozen=True)
class TrainConfig:
    dataset: str = "FD001"
    seed: int = 42
    split_seed: int = 42
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.002
    rul_cap: int = 125
    train_stride: int = 1
    validation_stride: int = 1
    validation_fraction: float = 0.2
    few_shot_fraction: float = 1.0
    num_workers: int = 0
    torch_threads: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    dataset: CMapssWindowDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_divisor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    units: list[np.ndarray] = []
    cycles: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["x"].to(device)).cpu().numpy() * target_divisor
            predictions.append(output)
            targets.append(batch["target"].numpy())
            units.append(batch["unit"].numpy())
            cycles.append(batch["cycle"].numpy())
    return tuple(np.concatenate(parts) for parts in (predictions, targets, units, cycles))  # type: ignore[return-value]


def clipped_metrics(targets: np.ndarray, predictions: np.ndarray, cap: int) -> dict[str, float]:
    return regression_metrics(targets, np.clip(predictions, 0.0, float(cap)))


def write_predictions(
    path: Path,
    units: np.ndarray,
    cycles: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    cap: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["unit", "cycle", "target", "prediction_raw", "prediction_clipped"])
        for row in zip(units, cycles, targets, predictions):
            unit, cycle, target, prediction = row
            writer.writerow(
                [int(unit), int(cycle), float(target), float(prediction), float(np.clip(prediction, 0, cap))]
            )


def write_prediction_svg(
    path: Path,
    units: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    cap: int,
) -> None:
    """Write a dependency-free SVG of official test endpoint predictions."""
    order = np.argsort(units)
    units = units[order].astype(float)
    targets = targets[order].astype(float)
    predictions = np.clip(predictions[order].astype(float), 0.0, float(cap))
    width, height = 1000, 520
    left, right, top, bottom = 70, 30, 45, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = float(units.min()), float(units.max())
    y_max = float(cap)

    def point(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / max(1.0, x_max - x_min) * plot_width
        py = top + (1.0 - y / y_max) * plot_height
        return px, py

    def polyline(values: np.ndarray) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(u, v) for u, v in zip(units, values)))

    grid = []
    for value in range(0, cap + 1, 25):
        _, y = point(x_min, value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#dddddd" stroke-width="1"/>'
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="24" text-anchor="middle" font-size="18">Official test endpoints: predicted vs actual RUL</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>
<polyline points="{polyline(targets)}" fill="none" stroke="#222222" stroke-width="2"/>
<polyline points="{polyline(predictions)}" fill="none" stroke="#d62728" stroke-width="2"/>
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14">Engine ID</text>
<text x="18" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height/2})">RUL</text>
<line x1="{width-235}" y1="20" x2="{width-200}" y2="20" stroke="#222222" stroke-width="2"/>
<text x="{width-195}" y="24" font-size="12">Actual</text>
<line x1="{width-125}" y1="20" x2="{width-90}" y2="20" stroke="#d62728" stroke-width="2"/>
<text x="{width-85}" y="24" font-size="12">Predicted</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, object]:
    config = TrainConfig(
        dataset=args.dataset,
        seed=args.seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_stride=args.train_stride,
        validation_stride=args.validation_stride,
        validation_fraction=args.validation_fraction,
        few_shot_fraction=args.few_shot_fraction,
        num_workers=args.num_workers,
        torch_threads=args.torch_threads,
    )
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)
    model_config = PatchTransformerConfig()
    seed_everything(config.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = prepare_cmapss_data(
        args.data_dir,
        config.dataset,
        seed=config.split_seed,
        validation_fraction=config.validation_fraction,
        few_shot_fraction=config.few_shot_fraction,
        window_size=model_config.window_size,
        train_stride=config.train_stride,
        validation_stride=config.validation_stride,
        rul_cap=config.rul_cap,
    )
    datasets = {name: getattr(bundle, name) for name in ("train", "val", "test")}
    source_hash = None
    if args.rebuilt_data_dir is not None:
        source_manifest = json.loads((args.rebuilt_data_dir / "manifest.json").read_text())
        if (source_manifest["dataset"] != config.dataset or source_manifest["split_seed"] != config.split_seed
                or source_manifest["train_sample_stride"] != config.train_stride
                or source_manifest["validation_sample_stride"] != config.validation_stride
                or config.few_shot_fraction != 1.0 or config.validation_fraction != 0.2):
            raise ValueError("temporal arguments do not match exported full-data protocol")
        datasets = {name: RebuiltWindowDataset(args.rebuilt_data_dir, name) for name in datasets}
        source_hash = file_sha256(args.rebuilt_data_dir / "manifest.json")
    train_loader = make_loader(datasets["train"], config.batch_size, True, config.seed, config.num_workers)
    val_loader = make_loader(datasets["val"], config.batch_size, False, config.seed, config.num_workers)
    test_loader = make_loader(datasets["test"], config.batch_size, False, config.seed, config.num_workers)

    model = PatchTransformer(model_config).to(device)
    initial_state_sha256 = state_digest(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    output_dir = args.output_dir or Path("artifacts") / "temporal" / config.dataset / f"seed{config.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    if source_hash is not None and checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite audited model: {output_dir}")
    history: list[dict[str, float | int]] = []
    best_val_rmse = math.inf
    started = time.time()

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
        record = {"epoch": epoch, "train_rmse": train_rmse, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(
            f"epoch={epoch:02d}/{config.epochs} train_rmse={train_rmse:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f}"
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "initial_state_sha256": initial_state_sha256,
                    "target_divisor": TARGET_DIVISOR,
                    "target_scale_status": TARGET_SCALE_STATUS,
                    "data_manifest_sha256": source_hash,
                    "model_config": model_config.to_dict(),
                    "train_config": asdict(config),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions, test_targets, test_units, test_cycles = predict(model, test_loader, device, checkpoint["target_divisor"])
    raw_metrics = regression_metrics(test_targets, test_predictions)
    test_metrics = clipped_metrics(test_targets, test_predictions, config.rul_cap)
    result: dict[str, object] = {
        "dataset": config.dataset,
        "initial_state_sha256": initial_state_sha256,
        "normalization": source_manifest.get("normalization", "global_minmax") if source_hash else "global_minmax",
        "target_divisor": checkpoint["target_divisor"],
        "target_scale_status": TARGET_SCALE_STATUS,
        "data_manifest_sha256": source_hash,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["val_metrics"],
        "test": test_metrics,
        "test_raw": raw_metrics,
        "test_prediction_count": int(len(test_predictions)),
        "elapsed_seconds": time.time() - started,
        "train_config": asdict(config),
        "model_config": model_config.to_dict(),
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
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if source_hash:
        scaler_path = args.rebuilt_data_dir / "scaler.json"
        if file_sha256(scaler_path) != source_manifest["scaler_sha256"]:
            raise ValueError("exported scaler checksum mismatch")
        shutil.copyfile(scaler_path, output_dir / "scaler.json")
    else:
        bundle.scaler.save(output_dir / "scaler.json")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="engine split seed, kept fixed when comparing model initialization seeds",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=1)
    parser.add_argument("--rebuilt-data-dir", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--few-shot-fraction", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="intra-op CPU threads; 0 keeps the PyTorch default",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
