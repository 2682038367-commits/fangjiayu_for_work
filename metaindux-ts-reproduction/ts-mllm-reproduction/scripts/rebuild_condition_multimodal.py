"""GPU feature-chain rebuild only; no temporal or TMAF training."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ts_mllm.rebuilt_data import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts/condition_multimodal_rebuild_seed42"
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    for dataset in ("FD002", "FD004"):
        data = root / f"artifacts/data_condition_minmax/split_seed42/{dataset}"
        data_hash = file_sha256(data / "manifest.json")
        mae = root / f"artifacts/mae_condition_minmax/{dataset}/stride50_seed42"
        alignment = root / f"artifacts/alignment_condition_minmax/{dataset}/stride50_seed42"
        cache = root / f"artifacts/qwen_condition_minmax/{dataset}/seed42"
        spectra = root / f"artifacts/spectrum_condition_minmax/{dataset}/stride50_seed42"
        stages = [
            ("mae", "pretrain_mae.py", ["--dataset", dataset, "--rebuilt-data-dir", data,
                "--output-dir", mae, "--device", "cuda", "--epochs", 15, "--batch-size", 128,
                "--learning-rate", 0.001], mae / "result.json"),
            ("alignment", "train_svlma_alignment.py", ["--dataset", dataset, "--rebuilt-data-dir", data,
                "--vision-checkpoint", mae / "best.pt", "--output-dir", alignment], alignment / "result.json"),
            ("cache", "cache_qwen_multimodal.py", ["--dataset", dataset, "--rebuilt-data-dir", data,
                "--vision-checkpoint", mae / "best.pt", "--alignment-checkpoint", alignment / "best.pt",
                "--output-dir", cache, "--spectrum-output-dir", spectra, "--train-stride", 50,
                "--validation-stride", 50, "--inference-batch-size", 16], cache / "manifest.json")]
        for name, script, flags, completed in stages:
            if not completed.exists():
                print(f"{dataset}/{name}: GPU start", flush=True)
                with (destination / f"{dataset}_{name}.log").open("a") as handle:
                    subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, flags)],
                                   cwd=root, stdout=handle, stderr=subprocess.STDOUT, check=True)
            record = json.loads(completed.read_text())
            assert record["dataset"] == dataset and record["data_manifest_sha256"] == data_hash
            print(f"{dataset}/{name}: complete and source checked", flush=True)
        manifest = json.loads((cache / "manifest.json").read_text())
        assert manifest["vision_checkpoint_sha256"] == file_sha256(mae / "best.pt")
        assert manifest["alignment_checkpoint_sha256"] == file_sha256(alignment / "best.pt")
        alignment_result = json.loads((alignment / "result.json").read_text())
        assert alignment_result["vision_checkpoint_sha256"] == manifest["vision_checkpoint_sha256"]
        for filename, expected in manifest["cache_file_sha256"].items():
            assert file_sha256(cache / filename) == expected
        spectrum_manifest = json.loads((spectra / "manifest.json").read_text())
        assert spectrum_manifest["data_manifest_sha256"] == data_hash
        assert spectrum_manifest["alignment_checkpoint_sha256"] == manifest["alignment_checkpoint_sha256"]
        spectrum_checks = {}
        for filename, expected in spectrum_manifest["files"].items():
            assert file_sha256(spectra / filename) == expected
            images = np.load(spectra / filename, mmap_mode="r")
            split = Path(filename).stem
            count = manifest["splits"][split]["count"]
            assert images.shape == (count, 3, 114, 114) and images.dtype == np.float32
            assert np.isfinite(images).all()
            spectrum_checks[split] = {"shape": list(images.shape), "finite": True, "checksum": "passed"}
        reports[dataset] = {"data_manifest_sha256": data_hash, "mae": str(mae), "alignment": str(alignment),
                            "qwen_cache": str(cache), "spectra": str(spectra), "cache_audits": manifest["audits"] if "audits" in manifest else manifest.get("audit"),
                            "spectrum_checks": spectrum_checks, "provenance_and_cache_checksums": "passed"}
        (destination / "summary.json").write_text(json.dumps(reports, indent=2))
    print("Both condition-normalized feature chains complete; TMAF not trained", flush=True)


if __name__ == "__main__":
    main()
