"""Pure temporal Patch Transformer from the TS-MLLM architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchTransformerConfig:
    window_size: int = 40
    input_channels: int = 14
    patch_size: int = 4
    patch_stride: int = 1
    model_dim: int = 64
    attention_heads: int = 1
    encoder_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1

    @property
    def patch_count(self) -> int:
        # The paper pads one stride on the right and gives
        # N = (L - P) / S + 2. This is exact for its L=40, P=4, S=1 setup.
        return (self.window_size + self.patch_stride - self.patch_size) // self.patch_stride + 1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PatchTransformer(nn.Module):
    """Patch embedding, two-layer Transformer, mean pooling, linear RUL head."""

    def __init__(self, config: PatchTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatchTransformerConfig()
        cfg = self.config
        if cfg.window_size < cfg.patch_size:
            raise ValueError("window_size must be at least patch_size")
        if cfg.model_dim % cfg.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")

        self.patch_projection = nn.Linear(cfg.patch_size * cfg.input_channels, cfg.model_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, cfg.patch_count, cfg.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first disables PyTorch's nested-tensor fast path implicitly;
        # make that choice explicit to avoid a misleading runtime warning.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.encoder_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.regression_head = nn.Linear(cfg.model_dim, 1)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def make_patches(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, L, M]`` inputs into flattened overlapping patches."""
        cfg = self.config
        if inputs.ndim != 3 or inputs.shape[1:] != (cfg.window_size, cfg.input_channels):
            raise ValueError(
                f"expected [B, {cfg.window_size}, {cfg.input_channels}], got {tuple(inputs.shape)}"
            )
        # Replicate the final time step on the right as described in the paper.
        padded = torch.cat([inputs, inputs[:, -1:, :].expand(-1, cfg.patch_stride, -1)], dim=1)
        patches = padded.unfold(1, cfg.patch_size, cfg.patch_stride)
        # unfold produces [B, N, M, P]; flatten each patch in temporal-major order.
        return patches.permute(0, 1, 3, 2).contiguous().flatten(start_dim=2)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(self.make_patches(inputs))
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        return self.output_norm(self.encoder(tokens))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(inputs).mean(dim=1)
        return self.regression_head(pooled).squeeze(-1)
