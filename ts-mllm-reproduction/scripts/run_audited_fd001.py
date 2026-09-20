#!/usr/bin/env python3
"""Priority4/5: rebuilt spectra/full tokens, new temporal model, literal TMAF."""

import json
from pathlib import Path

import torch

from cache_qwen_multimodal import parse_args as cache_args
from ts_mllm.qwen_cache import run as cache_run
from ts_mllm.training import parse_args as temporal_args, train as train_temporal
from ts_mllm.training_tmaf import parse_args as tmaf_args, train as train_tmaf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Stage1: new spectra and full Qwen tokens", flush=True)
    cache = cache_run(cache_args())
    torch.cuda.empty_cache()
    source = root / "artifacts/data_window40_stride50/split_seed42/FD001"
    temporal_output = root / "artifacts/temporal_audited/FD001/stride50_seed42"
    args = temporal_args()
    args.device = "cuda"
    args.rebuilt_data_dir = source
    args.train_stride = args.validation_stride = 50
    args.output_dir = temporal_output
    print("Stage2: time branch, sample stride50, patch stride1", flush=True)
    temporal = train_temporal(args)
    args = tmaf_args()
    args.rebuilt_data_dir = source
    args.train_stride = args.validation_stride = 50
    args.cache_dir = root / "artifacts/qwen_audited/FD001/seed42"
    args.temporal_checkpoint = temporal_output / "best.pt"
    args.context_mode = "global_broadcast"
    args.freeze_temporal_epochs = 0
    args.temporal_learning_rate = None
    args.output_dir = root / "artifacts/tmaf_audited/FD001/global_broadcast_stride50_seed42"
    print("Stage3: literal global broadcast TMAF, uniform lr0.002", flush=True)
    tmaf = train_tmaf(args)
    summary = {"fidelity_status": "paper_explicit_parameters_plus_disclosed_assumptions", "dataset": "FD001",
               "cache_counts": {name: details["count"] for name, details in cache["splits"].items()},
               "temporal": temporal["test"], "tmaf": tmaf["test"],
               "remaining": ["alignment training/bridge, MAE and Qwen model choices remain assumptions", "global vector broadcasting gives uniform attention; global masked mean is an assumption", "FD002-FD004 final multimodal runs not completed"]}
    destination = root / "artifacts/audited_fd001"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
