"""Temporal-centric Multi-modal Attention Fusion (TMAF)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .model import PatchTransformer, PatchTransformerConfig


@dataclass(frozen=True)
class TMAFConfig:
    temporal: PatchTransformerConfig = field(default_factory=PatchTransformerConfig)
    llm_dim: int = 1024
    attention_key_dim: int = 64
    attention_output_dim: int = 32
    fusion_mlp_units: int = 512
    dropout: float = 0.5
    context_mode: str = "tokens"
    global_pooling: str = "mean"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["temporal"] = self.temporal.to_dict()
        return payload


class TemporalMultimodalAttentionFusion(nn.Module):
    """Use temporal patches as queries over frozen-Qwen output tokens.

    This implements equations (14)--(19) in the paper: temporal Query,
    LLM-derived Key/Value, scaled dot-product attention, then an unconstrained
    linear projection over the concatenated temporal and retrieved features.
    In global_broadcast mode all projected keys/values are identical: softmax
    is necessarily uniform, so Q/K cannot select context by temporal patch.
    This is a limitation of the paper's literal broadcast formulation, not a
    working patch-selective attention mechanism.
    """

    def __init__(self, config: TMAFConfig | None = None) -> None:
        super().__init__()
        self.config = config or TMAFConfig()
        cfg = self.config
        if cfg.context_mode not in {"tokens", "global_broadcast"}:
            raise ValueError("context_mode must be tokens or global_broadcast")
        if cfg.global_pooling not in {"mean", "last_text"}:
            raise ValueError("global_pooling must be mean or last_text")
        if cfg.context_mode != "global_broadcast" and cfg.global_pooling != "mean":
            raise ValueError("last_text pooling requires global_broadcast")
        temporal_dim = cfg.temporal.model_dim
        self.temporal = PatchTransformer(cfg.temporal)
        self.query_projection = nn.Linear(temporal_dim, cfg.attention_key_dim)
        self.key_projection = nn.Linear(cfg.llm_dim, cfg.attention_key_dim)
        self.value_projection = nn.Linear(cfg.llm_dim, cfg.attention_output_dim)
        # Equation (19) is an unconstrained linear fusion gate. It deliberately
        # has no sigmoid, allowing positive reinforcement and noise suppression.
        self.fusion_projection = nn.Linear(
            temporal_dim + cfg.attention_output_dim, temporal_dim
        )
        self.regression_head = nn.Sequential(
            nn.Linear(temporal_dim, cfg.fusion_mlp_units),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_mlp_units, 1),
        )

    def load_temporal_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.temporal.load_state_dict(state_dict)

    def fuse(
        self,
        inputs: torch.Tensor,
        llm_tokens: torch.Tensor,
        llm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if llm_tokens.ndim != 3 or llm_tokens.shape[-1] != self.config.llm_dim:
            raise ValueError(
                f"expected LLM tokens [B,T,{self.config.llm_dim}], got {tuple(llm_tokens.shape)}"
            )
        if len(llm_tokens) != len(inputs):
            raise ValueError("time-series and LLM token batch sizes differ")
        temporal = self.temporal.encode(inputs)
        semantic = llm_tokens.float()
        if llm_mask is not None:
            if llm_mask.shape != llm_tokens.shape[:2] or not llm_mask.any(dim=1).all():
                raise ValueError("invalid LLM token mask")
        if self.config.context_mode == "global_broadcast":
            # Literal reading of III-D. The global aggregation itself is not
            # specified: masked mean is an explicit reproduction assumption.
            # Repeating this single vector makes the attention below uniform.
            weights = torch.ones_like(semantic[..., :1]) if llm_mask is None else llm_mask.unsqueeze(-1).float()
            if self.config.global_pooling == "mean":
                global_context = (semantic * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
            else:
                valid = torch.ones(semantic.shape[:2], device=semantic.device, dtype=torch.bool) if llm_mask is None else llm_mask.bool().clone()
                valid[:, 0] = False  # Position zero is the visual prefix, not text.
                if not valid.any(dim=1).all():
                    raise ValueError("last_text pooling requires a valid text token")
                positions = torch.arange(semantic.shape[1], device=semantic.device).expand_as(valid)
                last = positions.masked_fill(~valid, -1).max(dim=1).values
                global_context = semantic[torch.arange(len(semantic), device=semantic.device), last].unsqueeze(1)
            semantic = global_context.expand(-1, temporal.shape[1], -1)
            llm_mask = None
        queries = self.query_projection(temporal)
        keys = self.key_projection(semantic)
        values = self.value_projection(semantic)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
            self.config.attention_key_dim
        )
        if llm_mask is not None:
            if llm_mask.shape != llm_tokens.shape[:2]:
                raise ValueError("LLM mask shape does not match token sequence")
            if not llm_mask.any(dim=1).all():
                raise ValueError("every sample must contain at least one valid LLM token")
            scores = scores.masked_fill(~llm_mask.bool().unsqueeze(1), -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, values)
        fused = self.fusion_projection(torch.cat([temporal, context], dim=-1))
        return fused, attention

    def forward(
        self,
        inputs: torch.Tensor,
        llm_tokens: torch.Tensor,
        llm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused, _ = self.fuse(inputs, llm_tokens, llm_mask)
        return self.regression_head(fused.mean(dim=1)).squeeze(-1)
