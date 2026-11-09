"""Single-variable target-scale experiment with explicit prediction units."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import prepare_cmapss_data
from .metrics import regression_metrics
from .model import PatchTransformer
from .qwen_cache import require_cuda
from .rebuilt_data import RebuiltWindowDataset, file_sha256
from .training import seed_everything, make_loader, predict, clipped_metrics, write_predictions, write_prediction_svg


def to_training_target(rul: torch.Tensor, divisor: float) -> torch.Tensor:
    if divisor <= 0:
        raise ValueError("target divisor must be positive")
    return rul / divisor


def to_cycles(predictions: np.ndarray, divisor: float) -> np.ndarray:
    if divisor <= 0:
        raise ValueError("target divisor must be positive")
    return predictions * divisor


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def statistics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "prediction_mean_cycles": float(np.mean(predictions)),
        "prediction_std_cycles": float(np.std(predictions)),
        "prediction_min_cycles": float(np.min(predictions)),
        "prediction_max_cycles": float(np.max(predictions)),
        "target_mean_cycles": float(np.mean(targets)),
        "target_std_cycles": float(np.std(targets)),
    }


def run_experiment(project_root: Path, output_root: Path) -> dict[str, object]:
    device = require_cuda()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output_root}")
    source = project_root / "artifacts/data_window40_stride50/split_seed42/FD001"
    datasets = {name: RebuiltWindowDataset(source, name) for name in ("train", "val", "test")}
    # Independent reconstruction audit precedes either training run.
    fresh = prepare_cmapss_data(project_root.parent / "data/CMAPSSData", "FD001", seed=42,
                               train_stride=50, validation_stride=50)
    for split, cached in datasets.items():
        base = getattr(fresh, split)
        assert len(base) == len(cached)
        for index in range(len(cached)):
            actual = cached[index]
            expected = base[index]
            for field in ("x", "target", "unit", "cycle"):
                torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
    output_root.mkdir(parents=True)
    experiment = {}
    for name, divisor in (("A_raw", 1.0), ("B_div125", 125.0)):
        seed_everything(42)
        loaders = {split: make_loader(dataset, 128, split == "train", 42, 0)
                   for split, dataset in datasets.items()}
        model = PatchTransformer().to(device)
        initial_hash = state_hash(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        loss_function = torch.nn.MSELoss()
        output_dir = output_root / name
        output_dir.mkdir()
        initial = {}
        for split in ("train", "val"):
            diagnostic_loader = make_loader(datasets[split], 128, False, 42, 0)
            p, y, _, _ = predict(model, diagnostic_loader, device)
            initial[split] = statistics(to_cycles(p, divisor), y)
            initial[split]["native_prediction_mean"] = float(p.mean())
            initial[split]["native_prediction_std"] = float(p.std())
        parameters = dict(model.named_parameters())
        tracked = ["regression_head.weight", "regression_head.bias", "patch_projection.weight",
                   "position_embedding", "encoder.layers.0.self_attn.in_proj_weight", "output_norm.weight"]
        history = []
        best = math.inf
        updates = 0
        first_batch = None
        for epoch in range(1, 31):
            model.train()
            squared = 0.0
            count = 0
            gradients = {key: [] for key in tracked}
            for batch in loaders["train"]:
                x = batch["x"].to(device)
                rul = batch["target"].to(device)
                targets = to_training_target(rul, divisor)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(x)
                loss = loss_function(predictions, targets)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                for key in tracked:
                    gradient = parameters[key].grad
                    if gradient is None or not torch.isfinite(gradient).all():
                        raise ValueError(f"missing/non-finite gradient: {key}")
                    gradients[key].append(float(gradient.norm()))
                if first_batch is None:
                    first_batch = {
                        "training_target_mean": float(targets.mean()),
                        "target_mean_cycles": float(rul.mean()),
                        "native_prediction_mean": float(predictions.detach().mean()),
                        "prediction_mean_cycles": float(predictions.detach().mean()) * divisor,
                        "loss_training_units": float(loss.detach()),
                        "gradient_l2": {key: gradients[key][-1] for key in tracked},
                    }
                optimizer.step()
                squared += float(loss.detach()) * len(rul)
                count += len(rul)
                updates += 1
            p, y, _, _ = predict(model, loaders["val"], device)
            cycles = to_cycles(p, divisor)
            metrics = clipped_metrics(y, cycles, 125)
            row = {"epoch": epoch, "updates": updates,
                   "train_mse_training_units": squared / count,
                   "train_rmse_cycles": math.sqrt(squared / count) * divisor,
                   "val_clipped": metrics, "val_raw": regression_metrics(y, cycles),
                   "val_statistics": statistics(cycles, y),
                   "gradient_l2_mean": {key: float(np.mean(values)) for key, values in gradients.items()}}
            history.append(row)
            if metrics["rmse"] < best:
                best = metrics["rmse"]
                torch.save({"model_state": model.state_dict(), "target_divisor": divisor,
                            "epoch": epoch, "val_metrics_cycles": metrics,
                            "model_config": model.config.to_dict(),
                            "data_manifest_sha256": file_sha256(source / "manifest.json"),
                            "initial_state_sha256": initial_hash}, output_dir / "best.pt")
            print(f"{name} epoch={epoch:02d}/30 train_rmse_cycles={row['train_rmse_cycles']:.4f} "
                  f"val_rmse_cycles={metrics['rmse']:.4f} val_prediction_std={cycles.std():.4f}", flush=True)
        state = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        result = {"target_divisor": divisor, "initial_state_sha256": initial_hash,
                  "initial_statistics": initial, "first_training_batch": first_batch,
                  "best_epoch": state["epoch"], "validation_cycles": state["val_metrics_cycles"],
                  "updates": updates, "checkpoint_output_units": "cycles" if divisor == 1 else "cycles/125"}
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        experiment[name] = result
    assert experiment["A_raw"]["initial_state_sha256"] == experiment["B_div125"]["initial_state_sha256"]
    # Decide using validation before reading either checkpoint's test predictions.
    selected = min(experiment, key=lambda name: experiment[name]["validation_cycles"]["rmse"])
    selection = {"selected": selected, "criterion": "validation clipped RMSE in original cycle units",
                 "test_not_used_for_selection": True}
    (output_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for name, result in experiment.items():
        state = torch.load(output_root / name / "best.pt", map_location=device, weights_only=False)
        model = PatchTransformer().to(device)
        model.load_state_dict(state["model_state"])
        loader = make_loader(datasets["test"], 128, False, 42, 0)
        p, y, unit, cycle = predict(model, loader, device)
        cycles = to_cycles(p, state["target_divisor"])
        if len(p) != 100 or not np.isfinite(cycles).all():
            raise ValueError("test prediction count/finite check failed")
        np.testing.assert_array_equal(y, datasets["test"].arrays["target"])
        result["test_cycles"] = clipped_metrics(y, cycles, 125)
        result["test_raw_cycles"] = regression_metrics(y, cycles)
        result["test_statistics"] = statistics(cycles, y)
        write_predictions(output_root / name / "test_predictions.csv", unit, cycle, y, cycles, 125)
        write_prediction_svg(output_root / name / "test_predictions.svg", unit, y, cycles, 125)
        (output_root / name / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = {"device": torch.cuda.get_device_name(0), "dataset": "FD001", "seed": 42,
              "data_manifest_sha256": file_sha256(source / "manifest.json"),
              "source_reconstruction_audit": "all x/target/unit/cycle rows matched",
              "initial_weights_identical": True, "epochs": 30, "batch_size": 128, "learning_rate": 0.002,
              "train_sample_stride": 50, "validation_sample_stride": 50, "patch_stride": 1,
              "assumption": "target scaling is not specified in the paper; B trains RUL/125 and restores cycles",
              "selection": selection, "runs": experiment}
    (output_root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selection": selection, "runs": {name: {"val": result['validation_cycles'], "test": result['test_cycles']} for name, result in experiment.items()}}, indent=2), flush=True)
    return report
