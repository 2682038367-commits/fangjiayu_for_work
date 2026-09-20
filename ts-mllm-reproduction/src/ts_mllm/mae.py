"""Small ViT and masked-autoencoder models for 114x114 spectrum images."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import PatchTransformer, PatchTransformerConfig


@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int = 114
    patch_size: int = 19
    input_channels: int = 3
    embed_dim: int = 128
    attention_heads: int = 4
    encoder_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.1

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size**2

    @property
    def patch_dim(self) -> int:
        return self.input_channels * self.patch_size**2

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SpectrumViTEncoder(nn.Module):
    """Compact ViT encoder with a 128-dimensional CLS representation."""

    def __init__(self, config: VisionTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisionTransformerConfig()
        cfg = self.config
        self.patch_embed = nn.Conv2d(
            cfg.input_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, cfg.patch_count + 1, cfg.embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.encoder_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if images.shape[1:] != (cfg.input_channels, cfg.image_size, cfg.image_size):
            raise ValueError(
                f"expected [B,{cfg.input_channels},{cfg.image_size},{cfg.image_size}], "
                f"got {tuple(images.shape)}"
            )
        return self.patch_embed(images).flatten(2).transpose(1, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokens(images)
        cls = self.cls_token.expand(len(images), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.position_embedding
        return self.norm(self.encoder(tokens))[:, 0]


class SpectrumMaskedAutoencoder(nn.Module):
    """MAE pretraining objective sharing its encoder with SpectrumViTEncoder."""

    def __init__(
        self,
        config: VisionTransformerConfig | None = None,
        mask_ratio: float = 0.75,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        self.encoder = SpectrumViTEncoder(config)
        self.config = self.encoder.config
        self.mask_ratio = mask_ratio
        cfg = self.config
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.decoder_position = nn.Parameter(
            torch.empty(1, cfg.patch_count + 1, cfg.embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=decoder_layers, enable_nested_tensor=False
        )
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim)
        self.reconstruction_head = nn.Linear(cfg.embed_dim, cfg.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        size = cfg.patch_size
        patches = images.unfold(2, size, size).unfold(3, size, size)
        return patches.permute(0, 2, 3, 1, 4, 5).contiguous().flatten(start_dim=3).flatten(1, 2)

    def forward(
        self,
        images: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        patches = self.encoder.patch_tokens(images)
        batch, count, dim = patches.shape
        keep_count = max(1, int(round(count * (1.0 - self.mask_ratio))))
        noise = torch.rand(batch, count, device=images.device, generator=generator)
        shuffle = noise.argsort(dim=1)
        restore = shuffle.argsort(dim=1)
        keep = shuffle[:, :keep_count]
        visible = torch.gather(
            patches + self.encoder.position_embedding[:, 1:],
            1,
            keep.unsqueeze(-1).expand(-1, -1, dim),
        )
        cls = (self.encoder.cls_token + self.encoder.position_embedding[:, :1]).expand(
            batch, -1, -1
        )
        encoded = self.encoder.norm(self.encoder.encoder(torch.cat([cls, visible], dim=1)))

        masked = self.mask_token.expand(batch, count - keep_count, -1)
        shuffled_tokens = torch.cat([encoded[:, 1:], masked], dim=1)
        full_tokens = torch.gather(
            shuffled_tokens,
            1,
            restore.unsqueeze(-1).expand(-1, -1, dim),
        )
        decoder_input = torch.cat([encoded[:, :1], full_tokens], dim=1)
        decoder_input = decoder_input + self.decoder_position
        reconstruction = self.reconstruction_head(
            self.decoder_norm(self.decoder(decoder_input))[:, 1:]
        )
        mask = torch.ones(batch, count, device=images.device)
        mask[:, :keep_count] = 0
        mask = torch.gather(mask, 1, restore)
        target = self.patchify(images)
        loss_per_patch = (reconstruction - target).square().mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, reconstruction, mask


class TemporalViTRegressor(nn.Module):
    """Patch Transformer + ViT/MAE encoder using precomputed spectrum images."""

    def __init__(
        self,
        temporal_config: PatchTransformerConfig | None = None,
        vision_config: VisionTransformerConfig | None = None,
        fusion_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = PatchTransformer(temporal_config)
        self.visual = SpectrumViTEncoder(vision_config)
        self.fusion = nn.Sequential(
            nn.Linear(self.temporal.config.model_dim + self.visual.config.embed_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, inputs: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal.encode(inputs).mean(dim=1)
        visual = self.visual(images)
        return self.fusion(torch.cat([temporal, visual], dim=-1)).squeeze(-1)
