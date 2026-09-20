#!/usr/bin/env python3
"""Cache fixed RP/STFT/CWT images from a trained spectrum transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_mllm.data import prepare_cmapss_data
from ts_mllm.vision import TemporalVisualRegressor


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--data-dir", type=Path, default=project_root.parent / "data" / "CMAPSSData")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--validation-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint or Path("artifacts") / "temporal_visual" / args.dataset / f"seed{args.seed}" / "best.pt"
    output_dir = args.output_dir or Path("artifacts") / "spectrum_cache" / args.dataset / f"split_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_cmapss_data(
        args.data_dir, args.dataset, seed=args.seed,
        train_stride=args.train_stride, validation_stride=args.validation_stride,
    )
    model = TemporalVisualRegressor().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    shapes: dict[str, list[int]] = {}
    for split_name, dataset in (("train", data.train), ("val", data.val), ("test", data.test)):
        path = output_dir / f"{split_name}.npy"
        cache = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(dataset), 3, 114, 114),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                images = model.spectrum(batch["x"].to(device))
                values = images.mul(255).round().to(torch.uint8).cpu().numpy()
                cache[offset : offset + len(values)] = values
                offset += len(values)
        cache.flush()
        shapes[split_name] = list(cache.shape)
        print(f"cached {split_name}: {cache.shape} -> {path}", flush=True)

    manifest = {
        "dataset": args.dataset,
        "split_seed": args.seed,
        "train_stride": args.train_stride,
        "validation_stride": args.validation_stride,
        "source_checkpoint": str(checkpoint_path),
        "dtype": "uint8",
        "scale": 255,
        "channel_order": ["RP", "STFT", "Morlet-CWT"],
        "shapes": shapes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
