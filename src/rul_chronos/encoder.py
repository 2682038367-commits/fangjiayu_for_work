from __future__ import annotations

from typing import Protocol

import torch


def sensor_group_ids(batch_size: int, num_sensors: int, device: torch.device | str) -> torch.Tensor:
    """Assign every sensor row from one multivariate sample to the same Chronos group."""
    if batch_size < 1 or num_sensors < 1:
        raise ValueError("batch_size and num_sensors must both be positive")
    return torch.arange(batch_size, device=device).repeat_interleave(num_sensors)


def chronos_special_token_indices(
    num_context_patches: int,
    use_reg_token: bool,
    num_output_patches: int = 1,
) -> tuple[int | None, int, int]:
    """Return REG index, first future-query index, and expected encoder sequence length."""
    if num_context_patches < 1 or num_output_patches < 1:
        raise ValueError("context and output patch counts must both be positive")
    reg_index = num_context_patches if use_reg_token else None
    future_query_index = num_context_patches + int(use_reg_token)
    expected_length = future_query_index + num_output_patches
    return reg_index, future_query_index, expected_length


class Encoder(Protocol):
    embedding_dim: int

    def embed(self, context: torch.Tensor) -> torch.Tensor: ...


class Chronos2Encoder:
    """Frozen multivariate Chronos-2 encoder with configurable token pooling.

    Chronos-2 represents each variate as a row and connects rows that share the
    same group ID through group attention. For a batch [B, N, T], rows belonging
    to one sample therefore receive the same group ID. Supported representations
    are the last valid context patch, the mean over valid context patches, and
    the checkpoint's optional REG token.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "auto",
        dtype: str = "auto",
        representation: str = "last_valid_patch",
    ) -> None:
        try:
            from chronos import Chronos2Pipeline
        except ImportError as exc:
            raise RuntimeError("install the project dependencies before extracting Chronos-2 embeddings") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype: torch.dtype | str
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch_dtype,
        )
        self.model = self.pipeline.model
        self.model.eval()
        self.model.requires_grad_(False)
        self.embedding_dim = int(self.model.model_dim)
        self.device = self.model.device
        self.representation = representation
        if representation not in {"last_valid_patch", "valid_patch_mean", "reg_token", "future_query"}:
            raise ValueError(f"unsupported Chronos-2 representation: {representation}")

    @torch.inference_mode()
    def embed(self, context: torch.Tensor) -> torch.Tensor:
        return self.embed_variants(context, (self.representation,))[self.representation]

    @torch.inference_mode()
    def embed_variants(
        self,
        context: torch.Tensor,
        representations: tuple[str, ...] = (
            "last_valid_patch", "valid_patch_mean", "reg_token", "future_query"
        ),
    ) -> dict[str, torch.Tensor]:
        if context.ndim != 3:
            raise ValueError(f"context must have shape [B, N, T], got {tuple(context.shape)}")
        supported = {"last_valid_patch", "valid_patch_mean", "reg_token", "future_query"}
        unknown = set(representations) - supported
        if unknown:
            raise ValueError(f"unsupported Chronos-2 representations: {sorted(unknown)}")
        batch_size, num_sensors, history_length = context.shape
        model_context_length = int(self.model.chronos_config.context_length)
        if history_length > model_context_length:
            context = context[..., -model_context_length:]
        flat = context.reshape(batch_size * num_sensors, -1).to(self.device, dtype=torch.float32)
        context_mask = torch.isfinite(flat).to(self.model.dtype)
        patched_mask = torch.nan_to_num(self.model.patch(context_mask), nan=0.0)
        valid_patch_mask = patched_mask.sum(dim=-1) > 0
        group_ids = sensor_group_ids(batch_size, num_sensors, self.device)
        encoder_outputs, _, _, num_context_patches = self.model.encode(
            context=flat,
            group_ids=group_ids,
            num_output_patches=1,
        )
        hidden = encoder_outputs.last_hidden_state
        if hidden is None:
            raise RuntimeError("Chronos-2 returned no encoder hidden state")
        reg_index, future_query_index, expected_length = chronos_special_token_indices(
            num_context_patches,
            bool(self.model.chronos_config.use_reg_token),
            num_output_patches=1,
        )
        if hidden.shape[1] != expected_length:
            raise RuntimeError(
                f"unexpected Chronos-2 encoder sequence length: {hidden.shape[1]} != {expected_length}"
            )
        if valid_patch_mask.shape != (batch_size * num_sensors, num_context_patches):
            raise RuntimeError("derived valid-patch mask does not match Chronos-2 encoder output")
        context_hidden = hidden[:, :num_context_patches, :]
        result: dict[str, torch.Tensor] = {}
        if "last_valid_patch" in representations:
            positions = torch.arange(num_context_patches, device=self.device).expand_as(valid_patch_mask)
            last_indices = positions.masked_fill(~valid_patch_mask, -1).max(dim=-1).values
            if (last_indices < 0).any():
                raise RuntimeError("Chronos-2 input contains a row without any valid context patch")
            row_indices = torch.arange(len(context_hidden), device=self.device)
            last = context_hidden[row_indices, last_indices]
            result["last_valid_patch"] = last.reshape(batch_size, num_sensors, self.embedding_dim).float().cpu()
        if "valid_patch_mean" in representations:
            weights = valid_patch_mask.unsqueeze(-1).to(context_hidden.dtype)
            mean = (context_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            result["valid_patch_mean"] = mean.reshape(batch_size, num_sensors, self.embedding_dim).float().cpu()
        if "reg_token" in representations:
            if reg_index is None:
                raise RuntimeError("this Chronos-2 checkpoint does not define a REG token")
            reg = hidden[:, reg_index, :]
            result["reg_token"] = reg.reshape(batch_size, num_sensors, self.embedding_dim).float().cpu()
        if "future_query" in representations:
            future = hidden[:, future_query_index, :]
            result["future_query"] = future.reshape(batch_size, num_sensors, self.embedding_dim).float().cpu()
        return result


class DeterministicMockEncoder:
    """Small deterministic encoder for smoke tests; not a scientific baseline."""

    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim

    def embed(self, context: torch.Tensor) -> torch.Tensor:
        clean = torch.nan_to_num(context)
        last = clean[..., -1:]
        mean = clean.mean(dim=-1, keepdim=True)
        std = clean.std(dim=-1, keepdim=True, unbiased=False)
        base = torch.cat((last, mean, std), dim=-1)
        repeats = (self.embedding_dim + base.shape[-1] - 1) // base.shape[-1]
        return base.repeat(1, 1, repeats)[..., : self.embedding_dim].float()
