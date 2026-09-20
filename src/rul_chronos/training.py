from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache import EmbeddingCacheDataset
from .metrics import phm_score, rmse
from .model import WideDeepRULAdapter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        prediction = model(
            batch["embeddings"].to(device),
            batch["sensors"].to(device),
            batch["regimes"].to(device),
        )
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _cache_on_device(dataset: EmbeddingCacheDataset, device: torch.device) -> dict[str, torch.Tensor]:
    """Materialize a frozen embedding split once instead of converting every sample every epoch."""
    indices = dataset.indices
    embeddings = dataset.embeddings if indices is None else dataset.embeddings[indices]
    sensors = dataset.sensors if indices is None else dataset.sensors[indices]
    regimes = dataset.regimes if indices is None else dataset.regimes[indices]
    targets = dataset.targets if indices is None else dataset.targets[indices]
    return {
        "embeddings": torch.tensor(embeddings, dtype=torch.float32, device=device),
        "sensors": torch.tensor(sensors, dtype=torch.float32, device=device),
        "regimes": torch.tensor(regimes, dtype=torch.float32, device=device),
        "target": torch.tensor(targets, dtype=torch.float32, device=device),
    }


def _dataloader_random_order(size: int, generator: torch.Generator) -> torch.Tensor:
    """Match DataLoader(RandomSampler)'s RNG consumption and yielded order."""
    # _BaseDataLoaderIter draws a worker base seed before RandomSampler starts.
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return torch.randperm(size, generator=generator)


def _finish_dataloader_random_order(size: int, generator: torch.Generator) -> None:
    # RandomSampler makes one final randperm and takes an empty remainder when
    # num_samples == len(dataset). Consuming it preserves the next epoch's order.
    torch.randperm(size, generator=generator)


@torch.inference_mode()
def _predict_cached(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(tensors["target"]), batch_size):
        region = slice(start, start + batch_size)
        predictions.append(
            model(tensors["embeddings"][region], tensors["sensors"][region], tensors["regimes"][region]).cpu()
        )
    return tensors["target"].cpu().numpy(), torch.cat(predictions).numpy()


def train_adapter(
    cache_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
    batch_size: int = 1024,
    epochs: int = 150,
    patience: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    compression_dim: int = 8,
    dropout: float = 0.2,
    deep_only: bool = False,
    device_name: str = "auto",
    min_prefix_length: int = 1,
) -> dict[str, float | int]:
    seed_everything(seed)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    train_data = EmbeddingCacheDataset(cache_dir, "train", min_prefix_length=min_prefix_length)
    val_data = EmbeddingCacheDataset(cache_dir, "val", min_prefix_length=min_prefix_length)
    if not len(train_data) or not len(val_data):
        raise ValueError(f"min_prefix_length={min_prefix_length} removes all train or validation samples")
    metadata = train_data.metadata
    model_kwargs = {
        "num_sensors": int(metadata["num_sensors"]),
        "embedding_dim": int(metadata["embedding_dim"]),
        "compression_dim": compression_dim,
        "num_regimes": int(metadata["num_regimes"]),
        "dropout": dropout,
        "deep_only": deep_only,
    }
    model = WideDeepRULAdapter(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = None
    val_loader = None
    train_tensors = None
    val_tensors = None
    if device.type == "cuda":
        # FD001 occupies about 1.4 GB in float32 and fits comfortably on an 8 GB GPU.
        # Keeping it resident removes repeated float16->float32 conversion and PCIe copies.
        train_tensors = _cache_on_device(train_data, device)
        val_tensors = _cache_on_device(val_data, device)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    best_rmse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    completed_epochs = 0
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        model.train()
        running_loss = 0.0
        seen = 0
        if train_tensors is not None:
            order = _dataloader_random_order(len(train_data), generator)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size].to(device)
                optimizer.zero_grad(set_to_none=True)
                target = train_tensors["target"][indices]
                prediction = model(
                    train_tensors["embeddings"][indices],
                    train_tensors["sensors"][indices],
                    train_tensors["regimes"][indices],
                )
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item() * len(target)
                seen += len(target)
            _finish_dataloader_random_order(len(train_data), generator)
            assert val_tensors is not None
            # A non-worker DataLoader iterator also consumes one value from the
            # default CPU generator, even with shuffle=False.
            torch.empty((), dtype=torch.int64).random_()
            val_target, val_prediction = _predict_cached(model, val_tensors, batch_size)
        else:
            assert train_loader is not None and val_loader is not None
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                target = batch["target"].to(device)
                prediction = model(
                    batch["embeddings"].to(device),
                    batch["sensors"].to(device),
                    batch["regimes"].to(device),
                )
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item() * len(target)
                seen += len(target)
            val_target, val_prediction = predict(model, val_loader, device)
        val_rmse = rmse(val_target, val_prediction)
        print(f"epoch={epoch:03d} train_mse={running_loss / seen:.5f} val_rmse={val_rmse:.5f}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    assert best_state is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_kwargs": model_kwargs,
            "best_val_rmse": best_rmse,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "seed": seed,
            "min_prefix_length": min_prefix_length,
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
        },
        output_path,
    )
    return {
        "best_val_rmse": best_rmse,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
    }


def evaluate_adapter(
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 1024,
    device_name: str = "auto",
) -> dict[str, float]:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = WideDeepRULAdapter(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    test_data = EmbeddingCacheDataset(cache_dir, "test")
    target, prediction = predict(model, DataLoader(test_data, batch_size=batch_size, shuffle=False), device)
    result = {"rmse": rmse(target, prediction), "score": phm_score(target, prediction)}
    prediction_path = Path(checkpoint_path).with_suffix(".predictions.json")
    prediction_path.write_text(
        json.dumps(
            {
                "metrics": result,
                "unit": np.asarray(test_data.units).astype(int).tolist(),
                "cycle": np.asarray(test_data.cycles).astype(int).tolist(),
                "target": target.tolist(),
                "prediction": prediction.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
