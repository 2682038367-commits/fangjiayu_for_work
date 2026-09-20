"""Dynamic prompts and visual-prefix utilities for the frozen Qwen branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import ACTIVE_SENSOR_COLUMNS


def _trend(delta: float, threshold: float = 0.03) -> str:
    if delta > threshold:
        return "rising"
    if delta < -threshold:
        return "falling"
    return "stable"


def build_dynamic_prompt(window: np.ndarray | torch.Tensor) -> str:
    """Describe one normalized 40x14 window without using its RUL label."""
    if isinstance(window, torch.Tensor):
        values = window.detach().cpu().numpy()
    else:
        values = np.asarray(window)
    if values.shape != (40, len(ACTIVE_SENSOR_COLUMNS)):
        raise ValueError(f"expected (40, 14), got {values.shape}")
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0)
    deltas = values[-1] - values[0]
    volatility = np.abs(np.diff(values, axis=0)).mean(axis=0)
    strongest = np.argsort(np.abs(deltas))[-4:][::-1]
    changes = ", ".join(
        f"{ACTIVE_SENSOR_COLUMNS[index].replace('sensor_', 'S')} "
        f"{_trend(float(deltas[index]))} delta={deltas[index]:+.3f}"
        for index in strongest
    )
    mean_summary = ", ".join(
        f"S{ACTIVE_SENSOR_COLUMNS[index].split('_')[1]}={means[index]:.3f}"
        for index in range(len(ACTIVE_SENSOR_COLUMNS))
    )
    return (
        "Task: estimate remaining useful life for a turbofan engine from the current "
        "40-cycle normalized sensor window. Use only observed degradation evidence. "
        f"Global sensor std={standard_deviations.mean():.3f}; "
        f"mean cycle-to-cycle volatility={volatility.mean():.3f}. "
        f"Largest observed changes: {changes}. "
        f"Current sensor means: {mean_summary}."
    )


class SpectrumTextProjector(nn.Module):
    """Map the 128-d MAE representation into Qwen's token embedding space."""

    def __init__(
        self, visual_dim: int = 128, qwen_dim: int = 1024,
        architecture: str = "linear",
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "legacy_mlp"}:
            raise ValueError(f"unknown projector architecture {architecture}")
        self.architecture = architecture
        self.network = nn.Linear(visual_dim, qwen_dim) if architecture == "linear" else nn.Sequential(
            nn.Linear(visual_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.network(visual_features)


class DomainKnowledgeEmbedding(nn.Module):
    """Equation (11), with the 96-d text features specified in Table II.

    The paper does not specify how these 96-d vectors are bridged to Qwen's
    native embedding width. This module does not silently invent that bridge.
    """

    def __init__(self, vocabulary_size: int, feature_dim: int = 96, max_length: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, feature_dim)
        self.position = nn.Parameter(torch.zeros(1, max_length, feature_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.position.shape[1]:
            raise ValueError("text indices must be [B,L] with L <= maximum length")
        return self.embedding(input_ids) + self.position[:, :input_ids.shape[1]]


class QwenTextBridge(nn.Module):
    """96-d DKE plus a linear native-Qwen bridge (explicit assumption A-DKE-01)."""

    def __init__(self, vocabulary_size: int, qwen_dim: int = 1024) -> None:
        super().__init__()
        self.dke = DomainKnowledgeEmbedding(vocabulary_size, feature_dim=96, max_length=512)
        self.bridge = nn.Linear(96, qwen_dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.bridge(self.dke(input_ids))


def compose_multimodal_embeddings(
    visual_features: torch.Tensor,
    input_ids: torch.Tensor,
    projector: SpectrumTextProjector,
    text_adapter: QwenTextBridge,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The actual Qwen input path: linear vision prefix + bridged 96-d DKE."""
    if projector.architecture != "linear":
        raise ValueError("audited SVLMA requires a linear visual projector")
    visual = projector(visual_features.float()).unsqueeze(1)
    text = text_adapter(input_ids)
    return torch.cat([visual, text], dim=1).to(dtype=dtype)


@dataclass(frozen=True)
class MultimodalCacheConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    max_text_tokens: int = 512
    cached_text_tokens: int = 512
    visual_dim: int = 128
    qwen_dim: int = 1024

    @property
    def cached_tokens(self) -> int:
        return self.cached_text_tokens + 1


def select_cache_tokens(
    hidden_states: torch.Tensor,
    text_attention_mask: torch.Tensor,
    cached_text_tokens: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep visual-prefix output plus the last valid text token states."""
    batch, _, hidden = hidden_states.shape
    output = hidden_states.new_zeros(batch, cached_text_tokens + 1, hidden)
    mask = torch.zeros(
        batch, cached_text_tokens + 1, dtype=torch.bool, device=hidden_states.device
    )
    output[:, 0] = hidden_states[:, 0]
    mask[:, 0] = True
    for row in range(batch):
        length = int(text_attention_mask[row].sum())
        keep = min(length, cached_text_tokens)
        if keep:
            output[row, 1 : keep + 1] = hidden_states[row, 1 + length - keep : 1 + length]
            mask[row, 1 : keep + 1] = True
    return output, mask
