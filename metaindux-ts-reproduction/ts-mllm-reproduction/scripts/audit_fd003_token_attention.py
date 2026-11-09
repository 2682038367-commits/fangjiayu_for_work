"""GPU-only read-only comparison of FD003 token vs broadcast attention."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ts_mllm.model import PatchTransformerConfig
from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/fd003_figure12_v1/seed42"
DATA = ROOT / "artifacts/data_window40_stride50/split_seed42/FD003"


def load_model(subdir: str, device: torch.device):
    run = BASE / subdir
    result = json.loads((run / "result.json").read_text())
    state = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = result["model_config"]
    model = TemporalMultimodalAttentionFusion(TMAFConfig(
        temporal=PatchTransformerConfig(**cfg["temporal"]),
        llm_dim=cfg["llm_dim"], attention_key_dim=cfg["attention_key_dim"],
        attention_output_dim=cfg["attention_output_dim"],
        fusion_mlp_units=cfg["fusion_mlp_units"], dropout=cfg["dropout"],
        context_mode=cfg["context_mode"], global_pooling=cfg["global_pooling"],
    )).to(device).eval()
    model.load_state_dict(state["model_state"])
    return result, model


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    x = torch.tensor(np.load(DATA / "val_x.npy")[:8], device=device)
    tokens = torch.tensor(np.load(BASE / "qwen/val.npy")[:8], device=device)
    mask = torch.tensor(np.load(BASE / "qwen/val_mask.npy")[:8], device=device).bool()
    report = {"device": torch.cuda.get_device_name(), "validation_samples": len(x), "modes": {}}
    loaded = {}
    for subdir in ("tmaf_batch32", "tmaf_tokens_batch32"):
        result, model = load_model(subdir, device)
        loaded[subdir] = result
        _, attention = model.fuse(x, tokens, mask)
        if result["model_config"]["context_mode"] == "tokens":
            valid = mask[:, None, :]
            expected_uniform = valid.float() / valid.float().sum(-1, keepdim=True)
            max_masked = float(attention.masked_select(~valid.expand_as(attention)).abs().max().detach())
        else:
            expected_uniform = torch.full_like(attention, 1 / attention.shape[-1])
            max_masked = None
        model.zero_grad(set_to_none=True)
        prediction = model(x, tokens, mask)
        prediction.square().mean().backward()
        report["modes"][result["model_config"]["context_mode"]] = {
            "attention_shape": list(attention.shape),
            "max_deviation_from_uniform": float((attention - expected_uniform).abs().max().detach()),
            "mean_l1_distance_between_first_and_last_temporal_query": float((attention[:, 0] - attention[:, -1]).abs().sum(-1).mean().detach()),
            "max_masked_attention": max_masked,
            "query_weight_grad_l2": float(model.query_projection.weight.grad.norm()),
            "key_weight_grad_l2": float(model.key_projection.weight.grad.norm()),
            "value_weight_grad_l2": float(model.value_projection.weight.grad.norm()),
        }
        if result["model_config"]["context_mode"] == "tokens":
            visual_mass = attention[:, :, 0].detach()
            valid_probability = attention.masked_select(mask[:, None, :].expand_as(attention)).clamp_min(1e-30)
            report["modes"]["tokens"].update({
                "visual_prefix_attention_mean": float(visual_mass.mean()),
                "visual_prefix_attention_min": float(visual_mass.min()),
                "visual_prefix_attention_max": float(visual_mass.max()),
                "top_token_visual_fraction": float((attention.argmax(-1) == 0).float().mean()),
                "attention_entropy_mean": float((-(attention.clamp_min(1e-30).log() * attention).sum(-1)).mean()),
                "valid_token_count_first_sample": int(mask[0].sum()),
            })
    old, new = loaded["tmaf_batch32"], loaded["tmaf_tokens_batch32"]
    assert old["initialization"]["applied_initial_state_sha256"] == new["initialization"]["applied_initial_state_sha256"]
    assert old["data_manifest_sha256"] == new["data_manifest_sha256"]
    assert old["qwen_cache_manifest_sha256"] == new["qwen_cache_manifest_sha256"]
    assert old["temporal_checkpoint_sha256"] == new["temporal_checkpoint_sha256"]
    assert all(old[k] == new[k] for k in ("epochs", "batch_size", "learning_rate", "temporal_learning_rate", "freeze_temporal_epochs", "seed", "split_seed", "token_mode", "prompt_profile"))
    report["protocol_match"] = "passed: same initialization, cache, data, temporal source, and training settings; context_mode alone differs"
    # Validate that the near-one-hot behavior is not an accident of the first
    # eight validation windows used for the gradient probe.
    _, token_model = load_model("tmaf_tokens_batch32", device)
    full_x = np.load(DATA / "val_x.npy", mmap_mode="r")
    full_tokens = np.load(BASE / "qwen/val.npy", mmap_mode="r")
    full_mask = np.load(BASE / "qwen/val_mask.npy", mmap_mode="r")
    top_indices = Counter()
    max_weights = []
    query_l1 = []
    visual_mass = []
    with torch.inference_mode():
        for start in range(0, len(full_x), 16):
            stop = min(start + 16, len(full_x))
            bx = torch.tensor(np.array(full_x[start:stop]), device=device)
            bt = torch.tensor(np.array(full_tokens[start:stop]), device=device)
            bm = torch.tensor(np.array(full_mask[start:stop]), device=device).bool()
            _, att = token_model.fuse(bx, bt, bm)
            top_indices.update(att.argmax(-1).flatten().cpu().tolist())
            max_weights.extend(att.max(-1).values.flatten().cpu().tolist())
            query_l1.extend((att[:, 0] - att[:, -1]).abs().sum(-1).cpu().tolist())
            visual_mass.extend(att[:, :, 0].flatten().cpu().tolist())
    report["full_validation_attention"] = {
        "samples": len(full_x), "patch_queries": len(full_x) * 38,
        "top_token_indices_and_counts": top_indices.most_common(10),
        "mean_max_token_weight": float(np.mean(max_weights)),
        "fraction_max_token_weight_above_0_99": float(np.mean(np.asarray(max_weights) > 0.99)),
        "mean_first_last_query_l1": float(np.mean(query_l1)),
        "mean_visual_prefix_weight": float(np.mean(visual_mass)),
    }
    (ROOT / "FD003_TOKEN_ATTENTION_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
