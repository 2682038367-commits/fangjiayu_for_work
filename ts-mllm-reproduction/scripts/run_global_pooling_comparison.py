"""Single-variable last-text vs completed mean experiment; reuse Qwen caches."""
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
    for ds in args.datasets:
        base = root / f"artifacts/global_knowledge_sequence_v2/{ds}/seed42"
        mean = json.loads((base / "tmaf/result.json").read_text())
        cache = Path(mean["qwen_cache"])
        data = Path(mean["rebuilt_data_dir"])
        temporal = Path(mean["temporal_checkpoint"])
        for path, key in ((data / "manifest.json", "data_manifest_sha256"),
                          (cache / "manifest.json", "qwen_cache_manifest_sha256"),
                          (temporal, "temporal_checkpoint_sha256")):
            if file_sha256(path) != mean[key]:
                raise ValueError(f"mean source changed: {path}")
        if mean.get("prompt_profile") != "operating_sequence_v2" or mean["model_config"].get("global_pooling", "mean") != "mean":
            raise ValueError("expected completed ordered-prompt mean baseline")
        output = base / "tmaf_last_text"
        if not (output / "result.json").exists():
            command = [sys.executable, str(root / "scripts/train_tmaf.py"),
                "--dataset", ds, "--rebuilt-data-dir", str(data), "--cache-dir", str(cache),
                "--temporal-checkpoint", str(temporal), "--output-dir", str(output),
                "--context-mode", "global_broadcast", "--global-pooling", "last_text",
                "--token-mode", "full", "--epochs", "30", "--batch-size", "128",
                "--learning-rate", "0.002", "--temporal-learning-rate", "0.002",
                "--freeze-temporal-epochs", "0", "--output-bias-init", "default",
                "--train-stride", "50", "--validation-stride", "50", "--seed", "42", "--split-seed", "42"]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)
        if args.dry_run:
            continue
        last = json.loads((output / "result.json").read_text())
        for key in ("data_manifest_sha256", "qwen_cache_manifest_sha256", "temporal_checkpoint_sha256",
                    "epochs", "batch_size", "learning_rate", "temporal_learning_rate",
                    "freeze_temporal_epochs", "seed", "split_seed", "target_divisor"):
            if last[key] != mean[key]:
                raise ValueError(f"not a single-variable comparison: {key}")
        if last["model_config"].get("global_pooling") != "last_text":
            raise ValueError("wrong completed pooling mode")
        if last["initialization"]["applied_initial_state_sha256"] != mean["initialization"]["applied_initial_state_sha256"]:
            raise ValueError("initial weights differ")
        subprocess.run([sys.executable, str(root / "scripts/audit_tmaf_scale_b.py"), "--output-dir", str(output)], cwd=root, check=True)
        print(json.dumps({"dataset": ds, "status": "paper_unspecified_replication_assumption",
            "mean_validation": mean["validation"], "last_text_validation": last["validation"],
            "mean_test": mean["test"], "last_text_test": last["test"],
            "selected_by_validation": "last_text" if last["validation"]["rmse"] < mean["validation"]["rmse"] else "mean"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
