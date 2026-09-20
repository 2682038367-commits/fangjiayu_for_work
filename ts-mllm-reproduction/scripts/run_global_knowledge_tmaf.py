"""Global sensor Min-Max + operating knowledge; explicit new experiment namespace."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from ts_mllm.rebuilt_data import file_sha256
from ts_mllm.knowledge_prompt import PROFILE, KNOWLEDGE_SHA256, PROMPT_TEMPLATE_SHA256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["FD001","FD002","FD003","FD004"], default=["FD002","FD004"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        data = root / f"artifacts/data_window40_stride50/split_seed42/{dataset}"
        data_hash = file_sha256(data / "manifest.json")
        base = root / f"artifacts/global_knowledge_sequence_v2/{dataset}/seed42"
        mae = root / f"artifacts/mae_pretrain_audited/{dataset}/stride50_seed42"
        alignment, cache, spectra, output = [base / name for name in ("alignment","qwen","spectrum","tmaf")]
        temporal = root / ("artifacts/target_scale/FD001/seed42_verified/B_div125/best.pt" if dataset == "FD001"
                           else f"artifacts/temporal_audited/{dataset}/stride50_B_seed42/best.pt")
        if not temporal.is_file():
            raise FileNotFoundError(temporal)
        common = ["--dataset", dataset, "--rebuilt-data-dir", str(data)]
        stages = [
            ("MAE", "pretrain_mae.py", [*common, "--output-dir", str(mae), "--device", "cuda"], mae / "result.json"),
            ("alignment", "train_svlma_alignment.py", [*common, "--vision-checkpoint", str(mae / "best.pt"),
                "--output-dir", str(alignment), "--prompt-profile", PROFILE], alignment / "result.json"),
            ("cache", "cache_qwen_multimodal.py", [*common, "--vision-checkpoint", str(mae / "best.pt"),
                "--alignment-checkpoint", str(alignment / "best.pt"), "--output-dir", str(cache),
                "--spectrum-output-dir", str(spectra), "--prompt-profile", PROFILE], cache / "manifest.json"),
            ("TMAF", "train_tmaf.py", [*common, "--cache-dir", str(cache), "--temporal-checkpoint", str(temporal),
                "--output-dir", str(output), "--train-stride", "50", "--validation-stride", "50",
                "--context-mode", "global_broadcast", "--token-mode", "full", "--epochs", "30", "--batch-size", "128",
                "--learning-rate", "0.002", "--temporal-learning-rate", "0.002", "--freeze-temporal-epochs", "0",
                "--output-bias-init", "default", "--seed", "42", "--split-seed", "42"], output / "result.json")]
        for name, script, flags, completed in stages:
            if completed.exists():
                record = json.loads(completed.read_text())
                if record["dataset"] != dataset or record["data_manifest_sha256"] != data_hash:
                    raise ValueError(f"wrong completed-stage source: {completed}")
                if name in ("alignment","cache") and (record.get("prompt_profile") != PROFILE or record.get("knowledge_sha256") != KNOWLEDGE_SHA256):
                    raise ValueError("completed knowledge version mismatch")
                if name in ("alignment", "cache") and record.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256:
                    raise ValueError("completed prompt template changed; use a new experiment version")
                if name == "TMAF" and (record["qwen_cache_manifest_sha256"] != file_sha256(cache / "manifest.json")
                                       or record["temporal_checkpoint_sha256"] != file_sha256(temporal)):
                    raise ValueError("completed TMAF source mismatch")
                print(f"{dataset}/{name}: retain completed stage", flush=True)
                continue
            command = [sys.executable, str(root / "scripts" / script), *flags]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=root, check=True)
        command = [sys.executable, str(root / "scripts/audit_tmaf_scale_b.py"), "--output-dir", str(output)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
