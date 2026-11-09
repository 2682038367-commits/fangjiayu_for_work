"""Single-variable final output bias experiment, not a paper requirement."""
import hashlib

import torch


def state_digest(state, excluded=()):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name not in excluded:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_output_bias(model, mode, training_mean_cycles, divisor=125.0):
    if mode not in {"default", "train_mean"}:
        raise ValueError("unknown output bias initialization")
    if divisor <= 0 or not 0 <= training_mean_cycles <= divisor:
        raise ValueError("invalid training mean or divisor")
    key = "regression_head.3.bias"
    before = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    if mode == "train_mean":
        with torch.no_grad():
            model.regression_head[-1].bias.fill_(training_mean_cycles / divisor)
    after = model.state_dict()
    changed = [n for n in before if not torch.equal(before[n], after[n].detach().cpu())]
    assert changed == ([key] if mode == "train_mean" and before[key].item() != training_mean_cycles / divisor else [])
    return {"mode": mode, "status": "replication_assumption_paper_unspecified",
            "training_mean_cycles": training_mean_cycles, "target_divisor": divisor,
            "default_bias_native": before[key].item(), "applied_bias_native": after[key].item(),
            "changed_state_tensors": changed,
            "default_initial_state_sha256": state_digest(before),
            "applied_initial_state_sha256": state_digest(after),
            "unchanged_except_final_bias_sha256": state_digest(after, (key,))}
