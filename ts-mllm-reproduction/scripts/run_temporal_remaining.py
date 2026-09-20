#!/usr/bin/env python3
"""Train FD002--FD004 concurrently after FD001 has passed acceptance."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads-per-run", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    running: list[tuple[str, subprocess.Popen[bytes], object]] = []

    for dataset in ("FD002", "FD003", "FD004"):
        output_dir = project_root / "artifacts" / "temporal" / dataset / f"seed{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (output_dir / "run.log").open("wb")
        command = [
            sys.executable,
            "-u",
            "-m",
            "ts_mllm.training",
            "--dataset",
            dataset,
            "--seed",
            str(args.seed),
            "--split-seed",
            str(args.split_seed),
            "--epochs",
            str(args.epochs),
            "--torch-threads",
            str(args.threads_per_run),
            "--output-dir",
            str(output_dir),
        ]
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        running.append((dataset, process, log_handle))
        print(f"started {dataset}: pid={process.pid}, log={output_dir / 'run.log'}", flush=True)

    failures = []
    for dataset, process, log_handle in running:
        return_code = process.wait()
        log_handle.close()  # type: ignore[attr-defined]
        print(f"finished {dataset}: exit_code={return_code}", flush=True)
        if return_code:
            failures.append(dataset)
    if failures:
        raise SystemExit(f"failed datasets: {', '.join(failures)}")


if __name__ == "__main__":
    main()
