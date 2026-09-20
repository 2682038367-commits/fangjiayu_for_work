from __future__ import annotations

import torch
from torch import nn


class WideDeepRULAdapter(nn.Module):
    """Paper equations (3)--(5): shared compression, flatten, then fusion MLP."""

    def __init__(
        self,
        num_sensors: int = 21,
        embedding_dim: int = 768,
        compression_dim: int = 8,
        num_regimes: int = 1,
        dropout: float = 0.2,
        deep_only: bool = False,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.embedding_dim = embedding_dim
        self.compression_dim = compression_dim
        self.num_regimes = num_regimes
        self.deep_only = deep_only

        self.compression = nn.Linear(embedding_dim, compression_dim)
        deep_features = num_sensors * compression_dim
        wide_features = 0 if deep_only else num_sensors + num_regimes
        self.regressor = nn.Sequential(
            nn.Linear(deep_features + wide_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        sensors: torch.Tensor | None = None,
        regimes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must have shape [B, N, D], got {tuple(embeddings.shape)}")
        if embeddings.shape[1:] != (self.num_sensors, self.embedding_dim):
            raise ValueError(
                f"expected [B, {self.num_sensors}, {self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        deep = torch.nn.functional.gelu(self.compression(embeddings)).flatten(start_dim=1)
        if self.deep_only:
            fused = deep
        else:
            if sensors is None or regimes is None:
                raise ValueError("sensors and regimes are required for full Wide & Deep fusion")
            fused = torch.cat((deep, sensors, regimes), dim=-1)
        return self.regressor(fused).squeeze(-1)

