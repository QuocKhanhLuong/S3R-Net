"""Explicit spectral state for S3R-Net."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .spectral_utils import band_limited_features


class SpectralStateInitializer(nn.Module):
    """Initialize spectral state `S_t` from a feature map."""

    def __init__(self, channels: int, num_bands: int = 4, state_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or channels)
        self.proj = nn.Sequential(
            nn.Linear(self.channels, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )

    def forward(self, feat: Tensor) -> Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(feat.shape)}")
        if feat.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {feat.shape[1]}")
        bands = band_limited_features(feat, self.num_bands)
        summary = bands.mean(dim=(-2, -1))
        B, K, C = summary.shape
        state = self.proj(summary.reshape(B * K, C)).reshape(B, K, self.state_dim)
        return state


class SpectralStateTransition(nn.Module):
    """Learned band-specific spectral-state transition."""

    def __init__(self, channels: int, num_bands: int = 4, state_dim: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or channels)
        self.summary_proj = nn.Sequential(
            nn.Linear(self.channels, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.band_embed = nn.Parameter(torch.zeros(self.num_bands, self.state_dim))
        self.retain_gate = nn.Sequential(
            nn.Linear(self.state_dim * 3, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(self.state_dim * 3, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.band_embed, std=0.02)

    def forward(self, feat: Tensor, state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if feat.ndim != 4:
            raise ValueError(f"Expected feat [B,C,H,W], got {tuple(feat.shape)}")
        if state.ndim != 3:
            raise ValueError(f"Expected state [B,K,D], got {tuple(state.shape)}")
        B, C, _, _ = feat.shape
        if C != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {C}")
        if state.shape[0] != B or state.shape[1] != self.num_bands or state.shape[2] != self.state_dim:
            raise ValueError(
                f"Expected state shape [{B},{self.num_bands},{self.state_dim}], got {tuple(state.shape)}"
            )

        bands = band_limited_features(feat, self.num_bands)
        summary = bands.mean(dim=(-2, -1))
        z = self.summary_proj(summary.reshape(B * self.num_bands, C)).reshape(B, self.num_bands, self.state_dim)
        band = self.band_embed.unsqueeze(0).expand(B, -1, -1)
        gate_input = torch.cat([state, z, band], dim=-1)
        flat = gate_input.reshape(B * self.num_bands, self.state_dim * 3)
        retain = self.retain_gate(flat).reshape(B, self.num_bands, self.state_dim)
        update = self.update_gate(flat).reshape(B, self.num_bands, self.state_dim)
        new_state = retain * state + update * z
        logs = {
            "state_norm": new_state.norm(dim=-1),
            "state_delta": (new_state - state).norm(dim=-1),
            "state_retain_gate": retain.mean(dim=-1),
            "state_update_gate": update.mean(dim=-1),
        }
        return new_state, logs


class StateGuidedModulation(nn.Module):
    """Modulate feature channels with aggregated spectral state."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        state_dim: int | None = None,
        state_modulation_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or channels)
        self.scale = float(state_modulation_scale)
        hidden = max(self.state_dim, self.channels)
        self.mlp = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.channels),
            nn.Tanh(),
        )

    def forward(self, feat: Tensor, state: Tensor) -> Tensor:
        if state.ndim != 3:
            raise ValueError(f"Expected state [B,K,D], got {tuple(state.shape)}")
        if state.shape[1] != self.num_bands or state.shape[2] != self.state_dim:
            raise ValueError(f"Expected state [B,{self.num_bands},{self.state_dim}], got {tuple(state.shape)}")
        pooled = state.mean(dim=1)
        modulation = self.mlp(pooled).view(feat.shape[0], self.channels, 1, 1)
        return feat * (1.0 + self.scale * modulation)


def detach_state_logs(logs: dict[str, Tensor]) -> dict[str, Any]:
    """Convert state log tensors into serializable per-band means."""
    return {key: [float(v) for v in value.detach().mean(dim=0).cpu()] for key, value in logs.items()}
