"""Sequential GPU FD002–004 workflow with separate artifacts and resumable stages."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from ts_mllm.rebuilt_data import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["FD002", "FD003", "FD004"], default=["FD002", "FD003", "FD004"])
    args = parser.parse_args()
    destination = root / "artifacts/multimodal_remaining_B_seed42"
    destination.mkdir(parents=True, exist_ok=True)
    progress = {}

    def stage(dataset, name, script, flags, completed, data_hash):
        key = f"{dataset}/{name}"
        if completed.exists():
            record = json.loads(completed.read_text())
            if name != "audit" and (record.get("dataset") != dataset or record.get("data_manifest_sha256") != data_hash):
                raise ValueError(f"completed stage has wrong provenance: {completed}")
            progress[key] = "reused_completed"
            print(f"{key}: reuse {completed}", flush=True)
        else:
            progress[key] = "running"
            (destination / "progress.json").write_text(json.dumps(progress, indent=2))
            log = destination / f"{dataset}_{name}.log"
            print(f"{key}: GPU start; log={log}", flush=True)
            with log.open("a") as handle:
                process = subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, flags)],
                                         cwd=root, stdout=handle, stderr=subprocess.STDOUT)
            if process.returncode:
                progress[key] = f"failed_exit_{process.returncode}"
                (destination / "progress.json").write_text(json.dumps(progress, indent=2))
                raise RuntimeError(f"{key} failed; inspect {log}; no CPU fallback")
            assert completed.exists(), completed
            progress[key] = "complete"
            print(f"{key}: complete", flush=True)
        (destination / "progress.json").write_text(json.dumps(progress, indent=2))

    for dataset in args.datasets:
        data = root / f"artifacts/data_window40_stride50/split_seed42/{dataset}"
        data_hash = file_sha256(data / "manifest.json")
        mae = root / f"artifacts/mae_pretrain_audited/{dataset}/stride50_seed42"
        alignment = root / f"artifacts/svlma_alignment/{dataset}/stride50_seed42"
        cache = root / f"artifacts/qwen_audited/{dataset}/seed42"
        spectra = root / f"artifacts/spectrum_audited/{dataset}/stride50_seed42"
        temporal = root / f"artifacts/temporal_audited/{dataset}/stride50_B_seed42"
        tmaf = root / f"artifacts/tmaf_audited/{dataset}/global_broadcast_stride50_B_seed42"
        stage(dataset, "mae", "pretrain_mae.py",
              ["--dataset", dataset, "--rebuilt-data-dir", data, "--output-dir", mae, "--device", "cuda",
               "--epochs", 15, "--batch-size", 128, "--learning-rate", 0.001, "--seed", 42, "--split-seed", 42], mae / "result.json", data_hash)
        stage(dataset, "alignment", "train_svlma_alignment.py",
              ["--dataset", dataset, "--vision-checkpoint", mae / "best.pt", "--output-dir", alignment], alignment / "result.json", data_hash)
        stage(dataset, "cache", "cache_qwen_multimodal.py",
              ["--dataset", dataset, "--rebuilt-data-dir", data, "--vision-checkpoint", mae / "best.pt",
               "--alignment-checkpoint", alignment / "best.pt", "--output-dir", cache,
               "--spectrum-output-dir", spectra, "--train-stride", 50, "--validation-stride", 50,
               "--inference-batch-size", 16], cache / "manifest.json", data_hash)
        common = ["--dataset", dataset, "--rebuilt-data-dir", data, "--train-stride", 50,
                  "--validation-stride", 50, "--epochs", 30, "--batch-size", 128,
                  "--learning-rate", 0.002, "--seed", 42, "--split-seed", 42]
        stage(dataset, "temporal", "train_temporal.py",
              [*common, "--device", "cuda", "--torch-threads", 4, "--output-dir", temporal], temporal / "result.json", data_hash)
        stage(dataset, "tmaf", "train_tmaf.py",
              [*common, "--cache-dir", cache, "--temporal-checkpoint", temporal / "best.pt", "--output-dir", tmaf,
               "--context-mode", "global_broadcast", "--token-mode", "full", "--temporal-learning-rate", 0.002,
               "--freeze-temporal-epochs", 0, "--output-bias-init", "default"], tmaf / "result.json", data_hash)
        stage(dataset, "audit", "audit_tmaf_scale_b.py", ["--output-dir", tmaf], tmaf / "output_audit.json", data_hash)
    results = {}
    for dataset in ("FD001", "FD002", "FD003", "FD004"):
        temporal = root / ("artifacts/target_scale/FD001/seed42_verified/summary.json" if dataset == "FD001"
                           else f"artifacts/temporal_audited/{dataset}/stride50_B_seed42/result.json")
        tmaf = root / f"artifacts/tmaf_audited/{dataset}/global_broadcast_stride50_B_seed42/result.json"
        if temporal.exists() and tmaf.exists():
            time_result = json.loads(temporal.read_text())
            results[dataset] = {"temporal": time_result["runs"]["B_div125"] if dataset == "FD001" else time_result,
                                "tmaf": json.loads(tmaf.read_text())}
    summary = {"target_divisor": 125.0, "output_bias_init": "default", "seed": 42, "split_seed": 42,
               "stability_not_evaluated": True, "fidelity_status": "paper_explicit_settings_plus_disclosed_assumptions",
               "caveat": "FD001 reused compact MAE pretrained under legacy stride1 protocol; FD002–004 MAE now pretrained on audited stride50 windows",
               "results": results}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Requested GPU workflow finished; " + str(destination / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
