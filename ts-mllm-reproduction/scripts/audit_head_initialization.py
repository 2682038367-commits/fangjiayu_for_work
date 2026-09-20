"""Read-only GPU model audit; writes a new diagnostic report, never weights."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ts_mllm.qwen_dataset import CachedQwenDataset
from ts_mllm.rebuilt_data import RebuiltWindowDataset, file_sha256
from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.training import seed_everything
from ts_mllm.training_tmaf import require_cuda


def layer_comparison(model):
    first, second = model.temporal.encoder.layers
    left, right = dict(first.named_parameters()), dict(second.named_parameters())
    return {"all_initial_values_equal": all(torch.equal(left[n], right[n]) for n in left),
            "any_shared_parameter_storage": any(left[n].data_ptr() == right[n].data_ptr() for n in left),
            "different_parameter_tensors": sum(not torch.equal(left[n], right[n]) for n in left),
            "parameter_tensors_per_layer": len(left)}


def moments(tensor):
    tensor = tensor.detach().float()
    return {"mean": float(tensor.mean()), "std_population": float(tensor.std(unbiased=False)),
            "min": float(tensor.min()), "max": float(tensor.max()), "finite": bool(torch.isfinite(tensor).all())}


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "artifacts/head_initialization_audit/FD001/seed42"
    if (destination / "audit.json").exists():
        raise FileExistsError("refusing to overwrite completed audit")
    device = require_cuda()
    seed_everything(42)
    model = TemporalMultimodalAttentionFusion(TMAFConfig(context_mode="global_broadcast")).to(device)
    report = {"device": torch.cuda.get_device_name(0), "seed": 42,
              "architecture_changed": False, "weights_saved_or_modified": False,
              "fresh_temporal_layers": layer_comparison(model)}
    head = model.regression_head
    assert isinstance(head[0], nn.Linear) and (head[0].in_features, head[0].out_features) == (64, 512)
    assert isinstance(head[1], nn.GELU)
    assert isinstance(head[2], nn.Dropout) and head[2].p == 0.5
    assert isinstance(head[3], nn.Linear) and (head[3].in_features, head[3].out_features) == (512, 1)
    report["head"] = {"layers": [repr(layer) for layer in head], "pooling": "mean over patch dimension",
                      "paper_explicit_matches": {"fusion_mlp_units_512": True, "dropout_0_5": True},
                      "unspecified_assumptions": ["hidden-layer count", "GELU", "pooling", "bias", "scalar output implementation"]}
    report["fresh_head_parameters"] = {n: moments(p) for n, p in head.named_parameters()}
    fresh_head = {n: p.detach().cpu().clone() for n, p in head.named_parameters()}
    temporal_path = root / "artifacts/target_scale/FD001/seed42_verified/B_div125/best.pt"
    temporal = torch.load(temporal_path, map_location="cpu", weights_only=False)
    assert temporal["target_divisor"] == 125.0
    model.load_temporal_checkpoint(temporal["model_state"])
    report["B_checkpoint_temporal_layers"] = layer_comparison(model)
    report["temporal_checkpoint_sha256"] = file_sha256(temporal_path)
    base = RebuiltWindowDataset(root / "artifacts/data_window40_stride50/split_seed42/FD001", "train")
    dataset = CachedQwenDataset(base, root / "artifacts/qwen_audited/FD001/seed42", "train", token_mode="full")
    batch = next(iter(DataLoader(dataset, batch_size=128, shuffle=False)))
    x, tokens, mask = batch["x"].to(device), batch["llm_tokens"].to(device), batch["llm_mask"].to(device)
    model.eval()
    with torch.inference_mode():
        initial = model(x, tokens, mask)
    report["fresh_fusion_after_loading_B_eval"] = {"prediction_native": moments(initial),
                                                  "prediction_cycles": moments(initial * 125),
                                                  "target_cycles": moments(batch["target"])}
    trained_path = root / "artifacts/tmaf_audited/FD001/global_broadcast_stride50_B_seed42/best.pt"
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    model.load_state_dict(trained["model_state"])
    model.eval()
    with torch.inference_mode():
        fused, _ = model.fuse(x, tokens, mask)
        direct = model.regression_head(fused.mean(dim=1)).squeeze(-1)
        forward = model(x, tokens, mask)
        torch.testing.assert_close(direct, forward, rtol=0, atol=0)
    report["trained_head_forward_matches_explicit_mean_mlp"] = True
    report["trained_temporal_layers"] = layer_comparison(model)
    report["trained_checkpoint_sha256"] = file_sha256(trained_path)
    report["trained_head_parameters"] = {n: moments(p) for n, p in model.regression_head.named_parameters()}
    report["trained_head_delta_l2"] = {
        n: float((p.detach().cpu() - fresh_head[n]).norm())
        for n, p in model.regression_head.named_parameters()
    }
    report["initialization_assumptions"] = {
        "linear_weights": "torch Linear default kaiming_uniform_(a=sqrt(5)), equivalent to U(-1/sqrt(fan_in),+1/sqrt(fan_in))",
        "linear_biases": "torch default uniform; MHA projection biases zero",
        "attention_in_projection": "torch default Xavier uniform",
        "layernorm": "weights one, biases zero",
        "temporal_positions": "trunc_normal std=0.02 with torch default bounds [-2,2] (not +/-2 std)",
        "transformer_layers": "deep-copied equal values, independent storage at construction",
        "TMAF_temporal": "load B validation-best checkpoint; no temporal reinitialization",
        "TMAF_fusion_head": "fresh torch default initialization; no output bias prior"}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
