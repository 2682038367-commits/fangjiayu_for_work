#!/usr/bin/env python3
"""Verify actual parameter updates, head outputs, and saved cycle-unit CSVs."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ts_mllm.model import PatchTransformer
from ts_mllm.qwen_cache import require_cuda
from ts_mllm.rebuilt_data import RebuiltWindowDataset
from ts_mllm.target_scale import state_hash
from ts_mllm.training import seed_everything


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/target_scale/FD001/seed42_verified"
    data = RebuiltWindowDataset(root / "artifacts/data_window40_stride50/split_seed42/FD001", "test")
    device = require_cuda()
    seed_everything(42)
    initial = PatchTransformer().to(device)
    initial_hash = state_hash(initial)
    inputs = torch.stack([data[i]["x"] for i in range(len(data))]).to(device)
    report = {}
    for name in ("A_raw", "B_div125"):
        state = torch.load(output / name / "best.pt", map_location=device, weights_only=False)
        assert state["initial_state_sha256"] == initial_hash
        model = PatchTransformer().to(device).eval()
        model.load_state_dict(state["model_state"])
        changes = {key: float((value - initial.state_dict()[key]).norm())
                   for key, value in model.state_dict().items()
                   if key in ("regression_head.weight", "regression_head.bias", "patch_projection.weight", "position_embedding")}
        assert all(delta > 0 for delta in changes.values())
        with torch.inference_mode():
            direct_head = model.regression_head(model.encode(inputs).mean(1)).squeeze(-1)
            forward = model(inputs)
            torch.testing.assert_close(direct_head, forward)
            cycles = forward.cpu().numpy() * state["target_divisor"]
        rows = list(csv.DictReader((output / name / "test_predictions.csv").open()))
        np.testing.assert_allclose(cycles, [float(row["prediction_raw"]) for row in rows], rtol=1e-5, atol=2e-5)
        np.testing.assert_array_equal([int(row["unit"]) for row in rows], data.arrays["unit"])
        np.testing.assert_array_equal([float(row["target"]) for row in rows], data.arrays["target"])
        report[name] = {"parameter_changes_l2": changes, "direct_head_matches_forward": True,
                        "csv_predictions_match_restored_cycles": True, "test_count": len(rows)}
    (output / "output_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
