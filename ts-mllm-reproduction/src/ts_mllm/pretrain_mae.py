"""Self-supervised masked reconstruction pretraining on cached spectrum images."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import prepare_cmapss_data
from .mae import SpectrumMaskedAutoencoder, VisionTransformerConfig
from .spectrum_cache import CachedSpectrumDataset
from .training import seed_everything
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .vision import SpectrumTransform


def mean_loss(model: SpectrumMaskedAutoencoder, loader: DataLoader, device: torch.device, spectrum=None) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device) if spectrum is None else spectrum(batch["x"].to(device))
            loss, _, _ = model(images)
            total += float(loss) * len(images)
            count += len(images)
    return total / count


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    spectrum = None
    data_hash = None
    rebuilt_dir = getattr(args, "rebuilt_data_dir", None)
    if rebuilt_dir is not None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("audited MAE pretraining requires GPU")
        manifest = json.loads((rebuilt_dir / "manifest.json").read_text())
        if (manifest["dataset"] != args.dataset or manifest["split_seed"] != args.split_seed
                or manifest["window_size"] != 40 or manifest["train_sample_stride"] != 50
                or manifest["validation_sample_stride"] != 50):
            raise ValueError("MAE arguments do not match audited window40/stride50 data")
        train_data = RebuiltWindowDataset(rebuilt_dir, "train")
        val_data = RebuiltWindowDataset(rebuilt_dir, "val")
        data_hash = file_sha256(rebuilt_dir / "manifest.json")
        spectrum = SpectrumTransform().to(device).eval().requires_grad_(False)
    else:
        data = prepare_cmapss_data(args.data_dir, args.dataset, seed=args.split_seed)
        train_data = CachedSpectrumDataset(data.train, args.cache_dir / "train.npy")
        val_data = CachedSpectrumDataset(data.val, args.cache_dir / "val.npy")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False)
    vision_config = VisionTransformerConfig()
    model = SpectrumMaskedAutoencoder(vision_config, mask_ratio=args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    output_dir = args.output_dir or Path("artifacts") / "mae_pretrain" / args.dataset / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best.pt").exists():
        raise FileExistsError(f"refusing to overwrite MAE: {output_dir}")
    best = math.inf
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            with torch.no_grad():
                images = batch["image"].to(device) if spectrum is None else spectrum(batch["x"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(images)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            count += len(images)
        train_loss = total / count
        val_loss = mean_loss(model, val_loader, device, spectrum)
        history.append({"epoch": epoch, "train_reconstruction_mse": train_loss, "val_reconstruction_mse": val_loss})
        print(f"epoch={epoch:02d}/{args.epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dataset": args.dataset,
                    "data_manifest_sha256": data_hash,
                    "spectrum_initial_state": spectrum.state_dict() if spectrum is not None else None,
                    "encoder_state": model.encoder.state_dict(),
                    "vision_config": vision_config.to_dict(),
                    "epoch": epoch,
                    "val_reconstruction_mse": val_loss,
                    "mask_ratio": args.mask_ratio,
                },
                output_dir / "best.pt",
            )

    result = {
        "dataset": args.dataset,
        "data_manifest_sha256": data_hash,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pretraining_assumptions": ["compact local MAE", "15 epochs, AdamW lr0.001 weight_decay0.05", "mask0.75", "uniform sensor fusion during MAE; learned subsequently during alignment", "validation random masks select best reconstruction loss"],
        "best_val_reconstruction_mse": best,
        "best_epoch": min(history, key=lambda row: row["val_reconstruction_mse"])["epoch"],
        "epochs": args.epochs,
        "mask_ratio": args.mask_ratio,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.time() - started,
        "vision_config": vision_config.to_dict(),
        "cache_dir": str(args.cache_dir) if spectrum is None else None,
        "rebuilt_data_dir": str(rebuilt_dir) if rebuilt_dir is not None else None,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/spectrum_cache/FD001/split_seed42"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rebuilt-data-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
