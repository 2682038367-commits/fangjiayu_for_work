"""Train/replay condition-normalized full TMAF without rebuilding features."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ts_mllm.rebuilt_data import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["FD002", "FD004"], default=["FD002", "FD004"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        data = root / f"artifacts/data_condition_minmax/split_seed42/{dataset}"
        cache = root / f"artifacts/qwen_condition_minmax/{dataset}/seed42"
        temporal = root / f"artifacts/temporal_condition_minmax/{dataset}/stride50_B_seed42/best.pt"
        output = root / f"artifacts/tmaf_condition_minmax/{dataset}/global_broadcast_stride50_B_seed42"
        for required in (data / "manifest.json", cache / "manifest.json", temporal):
            if not required.is_file():
                raise FileNotFoundError(required)
        cache_manifest = json.loads((cache / "manifest.json").read_text())
        data_hash = file_sha256(data / "manifest.json")
        if cache_manifest["dataset"] != dataset or cache_manifest["data_manifest_sha256"] != data_hash:
            raise ValueError("new cache/data provenance mismatch")
        command = [sys.executable, str(root / "scripts/train_tmaf.py"), "--dataset", dataset,
                   "--rebuilt-data-dir", str(data), "--cache-dir", str(cache),
                   "--temporal-checkpoint", str(temporal), "--output-dir", str(output),
                   "--train-stride", "50", "--validation-stride", "50", "--context-mode", "global_broadcast",
                   "--token-mode", "full", "--epochs", "30", "--batch-size", "128", "--learning-rate", "0.002",
                   "--temporal-learning-rate", "0.002", "--freeze-temporal-epochs", "0",
                   "--output-bias-init", "default", "--seed", "42", "--split-seed", "42", "--num-workers", "2"]
        completed = output / "result.json"
        if completed.exists():
            result = json.loads(completed.read_text())
            if (result["data_manifest_sha256"] != data_hash
                    or result["temporal_checkpoint_sha256"] != file_sha256(temporal)
                    or result["qwen_cache_manifest_sha256"] != file_sha256(cache / "manifest.json")):
                raise ValueError("completed result has different source; refusing reuse")
            print(f"{dataset}: existing result retained; replay audit only", flush=True)
        else:
            if (output / "best.pt").exists():
                raise FileExistsError(f"partial run present; use a separate version, do not overwrite: {output}")
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)
        audit = [sys.executable, str(root / "scripts/audit_tmaf_scale_b.py"), "--output-dir", str(output)]
        print(shlex.join(audit), flush=True)
        if not args.dry_run:
            subprocess.run(audit, cwd=root, check=True)


if __name__ == "__main__":
    main()
