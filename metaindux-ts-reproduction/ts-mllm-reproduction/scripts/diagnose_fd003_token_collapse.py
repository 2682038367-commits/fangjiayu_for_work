"""Read-only GPU audit of FD003 token attention before and after TMAF training."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ts_mllm.bias_initialization import state_digest
from ts_mllm.tmaf import TMAFConfig, TemporalMultimodalAttentionFusion
from ts_mllm.training import seed_everything

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/fd003_figure12_v1/seed42"
TEMPORAL = ROOT / "artifacts/fd003_batch_diagnostic/seed42/batch32/best.pt"
DATA = ROOT / "artifacts/data_window40_stride50/split_seed42/FD003"


def describe(model, x, tokens, mask):
    top = Counter()
    max_prob, entropy, query_change, logit_gap = [], [], [], []
    query_norm, key_norm_3, key_norm_other, cache_norm_3, cache_norm_other = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(x), 16):
            stop = min(start + 16, len(x))
            xx = torch.tensor(np.array(x[start:stop]), device="cuda")
            zz = torch.tensor(np.array(tokens[start:stop]), device="cuda").float()
            mm = torch.tensor(np.array(mask[start:stop]), device="cuda").bool()
            assert mm[:, 0].all() and mm[:, 3].all()
            temporal = model.temporal.encode(xx)
            q = model.query_projection(temporal)
            k = model.key_projection(zz)
            logits = q @ k.transpose(-2, -1) / math.sqrt(model.config.attention_key_dim)
            logits = logits.masked_fill(~mm[:, None, :], -torch.inf)
            prob = logits.softmax(-1)
            _, direct = model.fuse(xx, zz, mm)
            torch.testing.assert_close(prob, direct, rtol=0, atol=0)
            top.update(prob.argmax(-1).flatten().cpu().tolist())
            max_prob.extend(prob.max(-1).values.flatten().cpu().tolist())
            entropy.extend((-(prob * prob.clamp_min(1e-30).log()).sum(-1)).flatten().cpu().tolist())
            query_change.extend((prob[:, 0] - prob[:, -1]).abs().sum(-1).cpu().tolist())
            # Top-1 minus runner-up logit, independent of a chosen token index.
            logit_gap.extend((logits.topk(2, dim=-1).values[..., 0] - logits.topk(2, dim=-1).values[..., 1]).flatten().cpu().tolist())
            query_norm.extend(q.norm(dim=-1).flatten().cpu().tolist())
            key_norm_3.extend(k[:, 3].norm(dim=-1).cpu().tolist())
            cache_norm_3.extend(zz[:, 3].norm(dim=-1).cpu().tolist())
            for i in range(len(zz)):
                selected = mm[i].clone(); selected[3] = False
                key_norm_other.append(float(k[i, selected].norm(dim=-1).median()))
                cache_norm_other.append(float(zz[i, selected].norm(dim=-1).median()))
    return {
        "n_samples": len(x), "n_queries": len(max_prob),
        "top_token_indices_and_counts": top.most_common(10),
        "mean_top_attention": float(np.mean(max_prob)),
        "fraction_top_attention_over_0_99": float(np.mean(np.asarray(max_prob) > 0.99)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_first_last_query_l1": float(np.mean(query_change)),
        "median_top1_runnerup_logit_gap": float(np.median(logit_gap)),
        "median_query_norm": float(np.median(query_norm)),
        "median_key3_norm": float(np.median(key_norm_3)),
        "median_other_valid_key_norm": float(np.median(key_norm_other)),
        "median_cache_token3_norm": float(np.median(cache_norm_3)),
        "median_other_valid_cache_norm": float(np.median(cache_norm_other)),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    result = json.loads((BASE / "tmaf_tokens_batch32/result.json").read_text())
    cfg = TMAFConfig(context_mode="tokens", global_pooling="mean")
    seed_everything(42)
    initial = TemporalMultimodalAttentionFusion(cfg).cuda().eval()
    temporal_state = torch.load(TEMPORAL, map_location="cpu", weights_only=False)
    initial.load_temporal_checkpoint(temporal_state["model_state"])
    assert state_digest(initial.state_dict()) == result["initialization"]["applied_initial_state_sha256"]
    trained = TemporalMultimodalAttentionFusion(cfg).cuda().eval()
    state = torch.load(BASE / "tmaf_tokens_batch32/best.pt", map_location="cuda", weights_only=False)
    trained.load_state_dict(state["model_state"])
    x = np.load(DATA / "val_x.npy", mmap_mode="r")
    z = np.load(BASE / "qwen/val.npy", mmap_mode="r")
    mask = np.load(BASE / "qwen/val_mask.npy", mmap_mode="r")
    assert len(x) == len(z) == len(mask) == 92
    valid_lengths = mask.sum(1)
    assert valid_lengths.min() >= 165 and valid_lengths.max() <= 166
    assert (np.abs(z[~mask]) < 1e-5).all()
    report = {"device": torch.cuda.get_device_name(), "initial_state_sha256_verified": True,
              "val_mask_and_padding_verified": True,
              "valid_token_range": [int(valid_lengths.min()), int(valid_lengths.max())],
              "before_training": describe(initial, x, z, mask),
              "best_checkpoint": describe(trained, x, z, mask),
              "explanation_limit": "Norm and logit statistics identify scale/softmax behavior, not a unique causal root. Token 3 is causally before dynamic statistics but after the visual prefix."}
    (ROOT / "FD003_TOKEN_COLLAPSE_DIAGNOSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
