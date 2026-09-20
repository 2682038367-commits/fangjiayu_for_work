"""Reconstruct paired initial states on GPU and select by validation only."""
import json
from pathlib import Path

import torch

from audit_tmaf_scale_b import main as replay
from ts_mllm.bias_initialization import initialize_output_bias
from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.training import seed_everything
from ts_mllm.training_tmaf import require_cuda


def main():
    root = Path(__file__).resolve().parents[1]
    original = root / "artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42"
    output = root / "artifacts/tmaf_bias_init/FD001/train_mean_seed42"
    a = json.loads((original / "result.json").read_text())
    b = json.loads((output / "result.json").read_text())
    for key in ("seed", "split_seed", "epochs", "batch_size", "learning_rate", "temporal_learning_rate",
                "freeze_temporal_epochs", "model_config", "data_manifest_sha256", "qwen_cache_manifest_sha256",
                "temporal_checkpoint_sha256", "target_divisor", "train_stride", "validation_stride", "token_mode"):
        assert a[key] == b[key], key
    seed_everything(42)
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast")).to(require_cuda())
    temporal = torch.load(root / a["temporal_checkpoint"], map_location="cpu", weights_only=False)
    model.load_temporal_checkpoint(temporal["model_state"])
    mean = b["initialization"]["training_mean_cycles"]
    default = initialize_output_bias(model, "default", mean)
    modified = initialize_output_bias(model, "train_mean", mean)
    assert modified == b["initialization"]
    assert default["unchanged_except_final_bias_sha256"] == modified["unchanged_except_final_bias_sha256"]
    assert modified["changed_state_tensors"] == ["regression_head.3.bias"]
    previous_audit = json.loads((root / "artifacts/head_initialization_audit/FD001/seed42/audit.json").read_text())
    assert previous_audit["fresh_head_parameters"]["3.bias"]["mean"] == default["default_bias_native"]
    selected = "train_mean" if b["validation"]["rmse"] < a["validation"]["rmse"] else "default"
    selection = {"selected": selected, "criterion": "validation clipped RMSE in cycle units",
                 "test_not_used_for_selection": True,
                 "A_validation_rmse": a["validation"]["rmse"], "B_validation_rmse": b["validation"]["rmse"],
                 "paired_configuration_checks": "passed",
                 "reconstructed_initial_state_difference": ["regression_head.3.bias"],
                 "limitation": "A full initial state was not saved; reconstructed using unchanged seed42 construction/load path; CUDA backward may be nondeterministic",
                 "A_reconstructed_initialization": default, "B_initialization": modified}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    replay(output)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
