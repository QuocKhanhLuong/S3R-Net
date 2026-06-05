"""Spectral helpers shared by S3R blocks, losses, and distillation."""

from __future__ import annotations

import torch
from torch import Tensor


def build_radial_frequency_masks(
    H: int,
    W: int,
    num_bands: int,
    device: torch.device | str,
) -> Tensor:
    """Return radial masks for an unshifted `torch.fft.rfft2` spectrum."""
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.rfftfreq(W, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / radius.max().clamp_min(1e-8)

    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device)
    masks = []
    for band in range(num_bands):
        left = edges[band]
        right = edges[band + 1]
        if band == num_bands - 1:
            mask = (radius >= left) & (radius <= right)
        else:
            mask = (radius >= left) & (radius < right)
        masks.append(mask.float())
    return torch.stack(masks, dim=0)


def band_limited_features(x: Tensor, num_bands: int) -> Tensor:
    """Return inverse-FFT band-limited feature maps `[B,K,C,H,W]`."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    B, C, H, W = x.shape
    X = torch.fft.rfft2(x.float(), norm="ortho")
    Wf = X.shape[-1]
    masks = build_radial_frequency_masks(H, W, num_bands, x.device).to(X.real.dtype)
    bands = []
    for band in range(num_bands):
        mask = masks[band].view(1, 1, H, Wf)
        bands.append(torch.fft.irfft2(X * mask, s=(H, W), norm="ortho"))
    return torch.stack(bands, dim=1)


def radial_band_amplitudes(x: Tensor, num_bands: int) -> Tensor:
    """Return mean rFFT amplitude per radial band as `[B,K]`."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [B,H,W], got {tuple(x.shape)}")
    B, C, H, W = x.shape
    X = torch.fft.rfft2(x.float(), norm="ortho")
    Wf = X.shape[-1]
    masks = build_radial_frequency_masks(H, W, num_bands, x.device).to(X.real.dtype)
    abs_x = X.abs()
    out = []
    for band in range(num_bands):
        mask = masks[band].view(1, 1, H, Wf)
        denom = mask.sum().clamp_min(1.0) * C
        out.append((abs_x * mask).sum(dim=(1, 2, 3)) / denom)
    return torch.stack(out, dim=1)


def num_groups(channels: int) -> int:
    """Choose a GroupNorm group count that divides `channels`."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
