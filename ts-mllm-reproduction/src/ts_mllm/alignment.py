"""Explicit assumption-based DKE/visual alignment, preserving paper dimensions."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .qwen_multimodal import QwenTextBridge, SpectrumTextProjector


@torch.no_grad()
def initialize_from_teacher(
    adapter: QwenTextBridge, teacher_embedding: nn.Module, training_ids: torch.Tensor
) -> None:
    """Train-vocabulary SVD initialization; never observes validation/test labels."""
    selected = teacher_embedding(torch.unique(training_ids)).float()
    if selected.shape[1] < 96:
        raise ValueError("teacher embedding width must be at least96")
    # Numeric/statistical prompts may use fewer than96 distinct subword IDs.
    # Complete the observed-token basis with orthogonal null-space directions
    # rather than changing the paper's required feature width.
    _, _, right = torch.linalg.svd(selected, full_matrices=True)
    basis = right[:96].T.contiguous()
    adapter.bridge.weight.copy_(basis)
    for start in range(0, len(adapter.dke.embedding.weight), 4096):
        stop = min(start + 4096, len(adapter.dke.embedding.weight))
        ids = torch.arange(start, stop, device=training_ids.device)
        adapter.dke.embedding.weight[start:stop].copy_(teacher_embedding(ids).float() @ basis)
    adapter.dke.position.zero_()


def alignment_losses(
    adapter: QwenTextBridge,
    projector: SpectrumTextProjector,
    ids: torch.Tensor,
    mask: torch.Tensor,
    visual: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = adapter(ids)
    weights = mask.unsqueeze(-1).float()
    text_mse = ((text - teacher).square() * weights).sum() / (weights.sum() * teacher.shape[-1])
    target_mean = (teacher * weights).sum(1) / weights.sum(1).clamp_min(1)
    visual_mse = (projector(visual) - target_mean).square().mean()
    return text_mse, visual_mse


def train_alignment(
    adapter: QwenTextBridge,
    projector: SpectrumTextProjector,
    teacher_embedding: nn.Module,
    splits: dict[str, dict[str, torch.Tensor]],
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.002,
    spectrum: nn.Module | None = None,
    vision_encoder: nn.Module | None = None,
) -> dict[str, object]:
    parameters = list(adapter.parameters()) + list(projector.parameters())
    if spectrum is not None:
        if vision_encoder is None:
            raise ValueError("vision encoder required for learnable online spectrum")
        parameters += list(spectrum.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        adapter.train()
        projector.train()
        totals = {"train": [0.0, 0.0, 0], "val": [0.0, 0.0, 0]}
        for split in ("train", "val"):
            data = splits[split]
            count = len(data["ids"])
            order = torch.randperm(count, device=data["ids"].device) if split == "train" else torch.arange(count, device=data["ids"].device)
            for index in order.split(batch_size):
                with torch.no_grad():
                    teacher = teacher_embedding(data["ids"][index]).float()
                with torch.set_grad_enabled(split == "train"):
                    visual = data["visual"][index] if spectrum is None else vision_encoder(spectrum(data["x"][index]))
                    text_loss, visual_loss = alignment_losses(
                        adapter, projector, data["ids"][index], data["mask"][index], visual, teacher
                    )
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        (text_loss + visual_loss).backward()
                        optimizer.step()
                totals[split][0] += float(text_loss.detach()) * len(index)
                totals[split][1] += float(visual_loss.detach()) * len(index)
                totals[split][2] += len(index)
        row = {"epoch": epoch}
        for split, (text, visual, count) in totals.items():
            row[f"{split}_text_mse"] = text / count
            row[f"{split}_visual_mse"] = visual / count
        history.append(row)
        val = row["val_text_mse"] + row["val_visual_mse"]
        print(f"alignment epoch={epoch:02d}/{epochs} val_text={row['val_text_mse']:.6f} val_visual={row['val_visual_mse']:.6f}", flush=True)
        if val < best:
            best = val
            torch.save({
                "text_adapter_state": adapter.state_dict(),
                "spectrum_state": spectrum.state_dict() if spectrum is not None else None,
                "projector_state": projector.state_dict(),
                "projector_architecture": "linear",
                "text_dim": 96,
                "max_text_tokens": 512,
                "epoch": epoch,
                "val_alignment_loss": val,
                "fidelity_status": "paper_dimensions_with_disclosed_alignment_assumptions",
                "assumptions": ["A-DKE-01: linear96-to-Qwen bridge", "A-DKE-02: training-vocabulary SVD initialization", "A-ALIGN-01: native frozen-Qwen token-embedding MSE distillation", "A-ALIGN-02: visual prefix to masked mean teacher-embedding MSE; equal loss weights", "A-ALIGN-03: freeze MAE/Qwen, train adapters with Adam; separate alignment stage"],
            }, output_dir / "best.pt")
    state = torch.load(output_dir / "best.pt", map_location=next(adapter.parameters()).device, weights_only=False)
    adapter.load_state_dict(state["text_adapter_state"])
    projector.load_state_dict(state["projector_state"])
    if spectrum is not None:
        spectrum.load_state_dict(state["spectrum_state"])
        spectrum.eval()
    adapter.eval()
    projector.eval()
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_epoch": state["epoch"], "val_alignment_loss": state["val_alignment_loss"], "assumptions": state["assumptions"]}
