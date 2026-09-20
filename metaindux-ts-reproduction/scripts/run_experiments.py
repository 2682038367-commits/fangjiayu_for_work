#!/usr/bin/env python3
"""Run the paper-aligned MetaIndux-TS experiment matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "upstream"


def gpu_runtime_metadata() -> dict:
    """Collect GPU details without making nvidia-smi a run dependency."""
    metadata = {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": None,
        "nvidia_driver": None,
    }
    if torch.cuda.is_available():
        metadata["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata["nvidia_driver"] = result.stdout.splitlines()[0].strip()
        except (FileNotFoundError, IndexError, subprocess.SubprocessError):
            pass
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paper_fd001.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-generated", action="store_true",
        help="Validate existing completed generation artifacts and run evaluation only.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a run only when its metadata says complete and all artifacts exist.",
    )
    return parser.parse_args()


def run_is_complete(metadata_path: Path, artifact_paths: list[Path]) -> bool:
    if not metadata_path.is_file() or not all(path.is_file() for path in artifact_paths):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "complete"


def validate_generated_artifacts(metadata_path, config, model_path, synth_path,
                                 loss_path, window_size):
    """Fail closed: never reuse incompatible or incomplete generation artifacts."""
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    for section in ("model", "diffusion", "training", "runtime"):
        if previous["config"][section] != config[section]:
            raise ValueError(f"cannot reuse {synth_path}: {section} config differs")
    history = np.loadtxt(loss_path, delimiter=",", skiprows=1, ndmin=2)
    epochs = int(config["training"]["epochs"])
    if history.shape != (epochs, 3) or not np.isfinite(history).all():
        raise ValueError(f"incomplete or invalid loss history: {loss_path}")
    if not np.array_equal(history[:, 0], np.arange(epochs)):
        raise ValueError(f"invalid epoch sequence: {loss_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not checkpoint or not all(torch.is_tensor(v) and torch.isfinite(v).all()
                                 for v in checkpoint.values()):
        raise ValueError(f"invalid checkpoint: {model_path}")
    with np.load(synth_path) as archive:
        data, labels = archive["data"], archive["label"]
        if (data.ndim != 3 or data.shape[0] == 0 or
                data.shape[1:] != (window_size, config["model"]["input_size"]) or
                labels.shape != (len(data), 1) or
                not np.isfinite(data).all() or not np.isfinite(labels).all()):
            raise ValueError(f"incomplete or invalid synthetic data: {synth_path}")
    return previous


def main() -> int:
    cli = parse_args()
    with cli.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = config["model"]
    diffusion = config["diffusion"]
    training = config["training"]
    runtime = config["runtime"]
    env = os.environ.copy()
    env["WANDB_MODE"] = runtime.get("wandb_mode", "offline")
    cache_root = PROJECT_ROOT / ".cache"
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    env["KEOPS_CACHE_FOLDER"] = str(cache_root / "keops")

    for dataset in config["experiment"]["datasets"]:
        for window_size in config["experiment"]["window_sizes"]:
            for seed in config["experiment"]["seeds"]:
                artifact_tag = config["experiment"].get("artifact_tag", "")
                suffix = f"_{artifact_tag}" if artifact_tag else ""
                stem = f"{dataset}_w{window_size}_seed{seed}{suffix}"
                model_path = PROJECT_ROOT / "checkpoints" / f"{stem}.pth"
                synth_path = PROJECT_ROOT / "outputs" / f"{stem}.npz"
                loss_path = PROJECT_ROOT / "logs" / f"{stem}_training_loss.csv"
                metrics_path = PROJECT_ROOT / "results" / "runs" / f"{stem}_metrics.json"
                metadata_path = PROJECT_ROOT / "results" / "runs" / f"{stem}_metadata.json"
                artifacts = [model_path, synth_path, loss_path, metrics_path]
                if cli.skip_completed and run_is_complete(metadata_path, artifacts):
                    print(f"skip completed: {stem}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "MainCondition.py",
                    "--dataset", dataset,
                    "--model_name", model["name"],
                    "--frequency_threshold", str(model.get("frequency_threshold", -1.0)),
                    "--frequency_mask_mode", model.get("frequency_mask_mode", "hard_random_quantile"),
                    "--frequency_threshold_init", str(model.get("frequency_threshold_init", 0.25)),
                    "--frequency_mask_temperature", str(model.get("frequency_mask_temperature", 0.1)),
                    "--state", runtime["state"],
                    "--window_size", str(window_size),
                    "--input_size", str(model["input_size"]),
                    "--T", str(diffusion["timesteps"]),
                    "--schedule_name", diffusion["beta_schedule"],
                    "--sample_type", diffusion["sample_type"],
                    "--epoch", str(training["epochs"]),
                    "--optimizer", training["optimizer"],
                    "--lr", str(training["learning_rate"]),
                    "--lr_schedule", training.get("lr_schedule", "warmup_cosine"),
                    "--batch-size", str(training["batch_size"]),
                    "--grad_clip", str(training["grad_clip"]),
                    "--seed", str(seed),
                    "--device", runtime["device"],
                    "--model_path", str(model_path),
                    "--syndata_path", str(synth_path),
                    "--loss_history_path", str(loss_path),
                    "--metrics_path", str(metrics_path),
                ]
                recovery_source = None
                if cli.reuse_generated and runtime["state"] == "all" and all(
                        p.is_file() for p in [model_path, synth_path, loss_path]):
                    recovery_source = validate_generated_artifacts(
                        metadata_path, config, model_path, synth_path, loss_path, window_size)
                    old_command = recovery_source.get("command", [])
                    # Only state may change; this also checks dataset, window and seed.
                    old_without_state = [v for i, v in enumerate(old_command)
                                         if v != "--state" and (i == 0 or old_command[i-1] != "--state")]
                    new_without_state = [v for i, v in enumerate(command)
                                         if v != "--state" and (i == 0 or command[i-1] != "--state")]
                    if old_without_state != new_without_state:
                        raise ValueError(f"cannot reuse {stem}: recorded command differs")
                    command[command.index("--state") + 1] = "eval"
                    print(f"reuse generated artifacts; evaluation only: {stem}", flush=True)
                print(" ".join(map(str, command)), flush=True)
                if not cli.dry_run:
                    started = time.time()
                    metadata = {
                        "status": "running",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "command": command,
                        "config": config,
                        "python": sys.version,
                        "platform": platform.platform(),
                        "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda,
                        **gpu_runtime_metadata(),
                    }
                    if recovery_source is not None:
                        metadata["recovery_source"] = recovery_source
                        metadata["reused_training_and_sampling"] = True
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    try:
                        subprocess.run(command, cwd=UPSTREAM, env=env, check=True)
                    except Exception:
                        metadata["status"] = "failed"
                        raise
                    else:
                        metadata["status"] = "complete"
                    finally:
                        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                        metadata["total_seconds"] = time.time() - started
                        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
