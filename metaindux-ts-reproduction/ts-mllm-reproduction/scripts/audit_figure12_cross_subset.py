"""Read-only provenance and matched-protocol audit for the Fig. 12 branch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/main_protocol_v1.json").read_text())

BASELINES = {
    "FD001": ROOT / "artifacts/fd001_mae_stride50_repair/seed42/tmaf",
    "FD002": ROOT / "artifacts/tmaf_condition_minmax/FD002/global_broadcast_stride50_B_seed42",
    "FD003": ROOT / "artifacts/fd003_batch_diagnostic/seed42/tmaf_batch32",
    "FD004": ROOT / "artifacts/tmaf_condition_minmax/FD004/global_broadcast_stride50_B_seed42",
}
EXPERIMENTS = {
    ds: ROOT / ("artifacts/fd003_figure12_v1/seed42" if ds == "FD003" else f"artifacts/figure12_v1/{ds}/seed42")
    for ds in BASELINES
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {"comparison_scope": "Fig.12 profile versus same-subset legacy A; FD003 uses batch32 diagnostic, others fixed batch128 main protocol", "datasets": {}}
    for ds in BASELINES:
        base = json.loads((BASELINES[ds] / "result.json").read_text())
        exp = EXPERIMENTS[ds]
        run = json.loads((exp / ("tmaf_batch32" if ds == "FD003" else "tmaf") / "result.json").read_text())
        cache = json.loads((exp / "qwen/manifest.json").read_text())
        old_cache_path = Path(base["qwen_cache"])
        old_cache = json.loads(((old_cache_path if old_cache_path.is_absolute() else ROOT / old_cache_path) / "manifest.json").read_text())
        alignment = json.loads((exp / "alignment/result.json").read_text())
        alignment_checkpoint = exp / "alignment/best.pt"
        data = ROOT / ("artifacts/data_condition_minmax/split_seed42" if ds in ("FD002", "FD004") else "artifacts/data_window40_stride50/split_seed42") / ds
        assert run["dataset"] == cache["dataset"] == alignment["dataset"] == ds
        assert run["prompt_profile"] == cache["prompt_profile"] == alignment["prompt_profile"] == "figure12_v1"
        assert alignment["prompt_template_sha256"] == cache["prompt_template_sha256"]
        assert alignment["knowledge_sha256"] == cache["knowledge_sha256"]
        assert cache["alignment_checkpoint_sha256"] == sha(alignment_checkpoint)
        assert cache["vision_checkpoint_sha256"] == old_cache["vision_checkpoint_sha256"]
        assert run["qwen_cache_manifest_sha256"] == sha(exp / "qwen/manifest.json")
        assert run["data_manifest_sha256"] == alignment["data_manifest_sha256"] == cache["data_manifest_sha256"] == base["data_manifest_sha256"] == sha(data / "manifest.json")
        assert run["temporal_checkpoint_sha256"] == base["temporal_checkpoint_sha256"]
        temporal_checkpoint = Path(run["temporal_checkpoint"])
        assert sha(temporal_checkpoint if temporal_checkpoint.is_absolute() else ROOT / temporal_checkpoint) == run["temporal_checkpoint_sha256"]
        base_model_config = dict(base["model_config"])
        base_model_config.setdefault("global_pooling", "mean")  # older audit omitted its default
        assert run["model_config"] == base_model_config
        assert run["initialization"]["applied_initial_state_sha256"] == base["initialization"]["applied_initial_state_sha256"]
        for key in ("seed", "split_seed", "epochs", "batch_size", "learning_rate", "temporal_learning_rate", "freeze_temporal_epochs", "token_mode", "target_divisor", "train_stride", "validation_stride"):
            assert run[key] == base[key], (ds, key)
        a = pd.read_csv(BASELINES[ds] / "test_predictions.csv")
        run_dir = exp / ("tmaf_batch32" if ds == "FD003" else "tmaf")
        b = pd.read_csv(run_dir / "test_predictions.csv")
        history = json.loads((run_dir / "history.json").read_text())
        assert len(history) == 30 and min(history, key=lambda row: row["val_rmse"])["epoch"] == run["best_epoch"]
        assert a[["unit", "cycle", "target"]].equals(b[["unit", "cycle", "target"]])
        for split in ("train", "val", "test"):
            assert cache["audit"][split]["mapping_and_full_text_length"] == "passed"
            assert cache["splits"][split]["valid_token_max"] <= 513
            assert np.array_equal(np.load(exp / "qwen" / f"{split}_target.npy"), np.load(data / f"{split}_target.npy"))
        report["datasets"][ds] = {
            "normalization": PROTOCOL["runs"][ds]["normalization"],
            "train_batch_size": run["batch_size"],
            "test_engines": len(a),
            "legacy_A": {k: base[k] for k in ("best_epoch", "validation", "test")},
            "figure12_v1": {k: run[k] for k in ("best_epoch", "validation", "test")},
            "validation_rmse_change": run["validation"]["rmse"] - base["validation"]["rmse"],
            "test_rmse_change_descriptive_only": run["test"]["rmse"] - base["test"]["rmse"],
            "audit": "passed: same data/temporal/model/initialization/training/test mapping; Fig12 alignment and cache hash chain valid",
        }
    (ROOT / "FIGURE12_CROSS_SUBSET_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
