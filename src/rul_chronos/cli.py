from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import extract_cache
from .data import prepare_splits
from .encoder import Chronos2Encoder, DeterministicMockEncoder
from .training import evaluate_adapter, train_adapter


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the Chronos-2 Wide & Deep RUL adapter")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="preprocess C-MAPSS and cache frozen Chronos-2 embeddings")
    _add_shared(extract)
    extract.add_argument("--data-dir", type=Path, required=True)
    extract.add_argument("--model-id", default="amazon/chronos-2")
    extract.add_argument("--device", default="auto")
    extract.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    extract.add_argument("--extract-batch-size", type=int, default=16)
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--mock-encoder", action="store_true", help="pipeline smoke test only; not scientific")

    train = commands.add_parser("train", help="train the lightweight adapter from cached embeddings")
    _add_shared(train)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-prefix-length", type=int, default=1)
    train.add_argument("--device", default="auto")
    train.add_argument("--deep-only", action="store_true")

    evaluate = commands.add_parser("evaluate", help="evaluate one adapter on official test endpoints")
    _add_shared(evaluate)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--batch-size", type=int, default=1024)
    evaluate.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = args.artifacts_dir / args.dataset
    if args.command == "extract":
        splits, normalizer, train_ids, val_ids = prepare_splits(args.data_dir, args.dataset, args.seed)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        normalizer.save(dataset_dir / "normalizer.json")
        (dataset_dir / "split.json").write_text(
            json.dumps({"seed": args.seed, "train_engine_ids": train_ids, "val_engine_ids": val_ids}, indent=2),
            encoding="utf-8",
        )
        encoder = DeterministicMockEncoder() if args.mock_encoder else Chronos2Encoder(args.model_id, args.device, args.dtype)
        for split, split_data in splits.items():
            extract_cache(split_data, encoder, dataset_dir, split, args.extract_batch_size)
        print(f"cached embeddings in {dataset_dir}")
    elif args.command == "train":
        suffix = "deep_only" if args.deep_only else "wide_deep"
        checkpoint = dataset_dir / f"adapter_{suffix}_seed{args.seed}.pt"
        result = train_adapter(
            dataset_dir,
            checkpoint,
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            min_prefix_length=args.min_prefix_length,
            deep_only=args.deep_only,
            device_name=args.device,
        )
        print(json.dumps({"checkpoint": str(checkpoint), **result}, indent=2))
    elif args.command == "evaluate":
        checkpoint = args.checkpoint or dataset_dir / "adapter_wide_deep_seed42.pt"
        result = evaluate_adapter(dataset_dir, checkpoint, args.batch_size, args.device)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
