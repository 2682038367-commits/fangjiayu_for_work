"""RP/STFT/CWT spectrum construction and the first visual baseline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

from .model import PatchTransformer, PatchTransformerConfig


class SpectrumTransform(nn.Module):
    """Convert a ``[B, 40, 14]`` sensor window into an RGB-like spectrum image.

    Channel 0 is a multivariate recurrence plot, channel 1 is a sensor-fused
    STFT magnitude map, and channel 2 is a sensor-fused Morlet CWT map. Sensor
    fusion weights are learned and normalized independently for STFT and CWT.
    """

    def __init__(
        self,
        input_channels: int = 14,
        image_size: int = 114,
        rp_threshold: float = 0.25,
        stft_n_fft: int = 16,
        stft_hop_length: int = 4,
        cwt_scales: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16),
        cwt_kernel_size: int = 31,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.rp_threshold = rp_threshold
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.cwt_scales = cwt_scales
        self.cwt_kernel_size = cwt_kernel_size
        self.stft_sensor_logits = nn.Parameter(torch.zeros(input_channels))
        self.cwt_sensor_logits = nn.Parameter(torch.zeros(input_channels))
        self.register_buffer("stft_window", torch.hann_window(stft_n_fft), persistent=False)
        self.register_buffer(
            "cwt_kernels",
            self._morlet_kernels(cwt_scales, cwt_kernel_size),
            persistent=False,
        )

    @staticmethod
    def _morlet_kernels(scales: tuple[int, ...], kernel_size: int) -> torch.Tensor:
        if kernel_size % 2 == 0:
            raise ValueError("cwt_kernel_size must be odd")
        center = kernel_size // 2
        time = torch.arange(-center, center + 1, dtype=torch.float32)
        kernels = []
        for scale in scales:
            scaled = time / float(scale)
            wavelet = torch.cos(5.0 * scaled) * torch.exp(-0.5 * scaled.square())
            wavelet = wavelet - wavelet.mean()
            wavelet = wavelet / wavelet.square().sum().sqrt().clamp_min(1e-8)
            kernels.append(wavelet)
        return torch.stack(kernels).unsqueeze(1)

    @staticmethod
    def _normalize(image: torch.Tensor) -> torch.Tensor:
        flat = image.flatten(start_dim=1)
        minimum = flat.min(dim=1).values[:, None, None]
        maximum = flat.max(dim=1).values[:, None, None]
        return (image - minimum) / (maximum - minimum).clamp_min(1e-6)

    def recurrence_plot(self, inputs: torch.Tensor) -> torch.Tensor:
        # Normalize each sensor within the window so no high-range channel
        # dominates the multivariate Euclidean distance.
        minimum = inputs.min(dim=1, keepdim=True).values
        maximum = inputs.max(dim=1, keepdim=True).values
        normalized = (inputs - minimum) / (maximum - minimum).clamp_min(1e-6)
        distance = torch.cdist(normalized, normalized) / math.sqrt(self.input_channels)
        return (distance <= self.rp_threshold).to(inputs.dtype)

    def stft_map(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, channels = inputs.shape
        series = inputs.transpose(1, 2).reshape(batch * channels, length)
        spectrum = torch.stft(
            series,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_n_fft,
            window=self.stft_window.to(device=inputs.device, dtype=inputs.dtype),
            center=True,
            return_complex=True,
        ).abs()
        spectrum = torch.log1p(spectrum).reshape(batch, channels, *spectrum.shape[-2:])
        weights = self.stft_sensor_logits.softmax(dim=0)
        return (spectrum * weights[None, :, None, None]).sum(dim=1)

    def cwt_map(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, channels = inputs.shape
        series = inputs.transpose(1, 2).reshape(batch * channels, 1, length)
        coefficients = functional.conv1d(
            series,
            self.cwt_kernels.to(device=inputs.device, dtype=inputs.dtype),
            padding=self.cwt_kernel_size // 2,
        ).abs()
        coefficients = torch.log1p(coefficients).reshape(
            batch, channels, len(self.cwt_scales), length
        )
        weights = self.cwt_sensor_logits.softmax(dim=0)
        return (coefficients * weights[None, :, None, None]).sum(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_channels:
            raise ValueError(f"expected [B, L, {self.input_channels}], got {tuple(inputs.shape)}")
        rp = self.recurrence_plot(inputs).unsqueeze(1)
        stft = self._normalize(self.stft_map(inputs)).unsqueeze(1)
        cwt = self._normalize(self.cwt_map(inputs)).unsqueeze(1)
        rp = functional.interpolate(rp, (self.image_size, self.image_size), mode="nearest")
        stft = functional.interpolate(
            stft, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        cwt = functional.interpolate(
            cwt, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return torch.cat([rp, stft, cwt], dim=1).clamp(0.0, 1.0)


class SpectrumCNNEncoder(nn.Module):
    """Lightweight CNN visual encoder used as the pre-MAE visual baseline."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(images).flatten(start_dim=1))


class TemporalVisualRegressor(nn.Module):
    """Temporal Patch Transformer plus spectrum CNN with late feature fusion."""

    def __init__(
        self,
        temporal_config: PatchTransformerConfig | None = None,
        visual_dim: int = 128,
        fusion_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.temporal = PatchTransformer(temporal_config)
        self.spectrum = SpectrumTransform(
            input_channels=self.temporal.config.input_channels,
            image_size=114,
        )
        self.visual = SpectrumCNNEncoder(visual_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.temporal.config.model_dim + visual_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal.encode(inputs).mean(dim=1)
        visual_features = self.visual(self.spectrum(inputs))
        return self.fusion(torch.cat([temporal_features, visual_features], dim=-1)).squeeze(-1)
