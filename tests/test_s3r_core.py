from __future__ import annotations

import math

import torch
from src.models.s3r import S3RMini, S3RNet
from src.models.s3r.losses import S3RLoss, boundary_map_from_mask
from src.models.s3r.s3r_blocks import SSRFullBlock, build_radial_frequency_masks
from src.models.s3r.s3r_state import (
    SpectralStateInitializer,
    SpectralStateTransition,
    StateGuidedModulation,
)


def test_radial_frequency_masks_cover_rfft_spectrum() -> None:
    masks = build_radial_frequency_masks(16, 20, num_bands=4, device="cpu")

    assert masks.shape == (4, 16, 11)
    assert torch.allclose(masks.sum(dim=0), torch.ones(16, 11))
    assert torch.all((masks == 0) | (masks == 1))
    assert torch.all(masks.sum(dim=(1, 2)) > 0)

    odd_masks = build_radial_frequency_masks(15, 17, num_bands=4, device="cpu")
    assert odd_masks.shape == (4, 15, 9)
    assert torch.allclose(odd_masks.sum(dim=0), torch.ones(15, 9))
    assert torch.all(odd_masks.sum(dim=(1, 2)) > 0)


def test_fft_band_reconstruction_identity() -> None:
    x = torch.randn(2, 3, 16, 20)
    X = torch.fft.rfft2(x, norm="ortho")
    masks = build_radial_frequency_masks(16, 20, num_bands=4, device=x.device).to(X.real.dtype)
    reconstructed = torch.zeros_like(x)
    for mask in masks:
        reconstructed = reconstructed + torch.fft.irfft2(X * mask.view(1, 1, 16, 11), s=(16, 20), norm="ortho")

    assert torch.mean(torch.abs(reconstructed - x)).item() < 1e-5


def test_ssr_full_block_returns_aux_dictionary_and_logs() -> None:
    block = SSRFullBlock(channels=8, num_bands=4)
    x = torch.randn(2, 8, 16, 16)
    boundary = torch.zeros(2, 1, 16, 16)
    boundary[:, :, 4:8, 4:8] = 1.0

    y, aux = block(x, boundary_mask=boundary, return_logs=True)

    assert y.shape == x.shape
    assert aux["gate_reg"].ndim == 0
    assert aux["hf_ratio_penalty"].ndim == 0
    logs = aux["logs"]
    for key in (
        "retain_gate_mean",
        "suppress_gate_mean",
        "update_gate_mean",
        "phase_coherence",
        "variance",
        "boundary_to_nonboundary_high_ratio",
        "update_budget_sum",
        "feature_norm",
        "delta_norm",
        "gamma_delta_norm",
        "residual_ratio",
        "gamma_raw",
        "gamma_effective",
        "relative_energy",
        "log_energy",
        "block_input_mean",
        "block_input_std",
        "block_output_mean",
        "block_output_std",
        "output_delta_mean",
        "output_delta_std",
        "gamma",
    ):
        assert key in logs


def test_ssr_gamma_zero_variant_is_identity_before_state_modulation() -> None:
    block = SSRFullBlock(channels=8, num_bands=4, block_variant="s3r_gamma0")
    x = torch.randn(2, 8, 16, 16)

    y, aux = block(x, return_logs=True)

    assert torch.allclose(y, x, atol=1e-5)
    assert aux["logs"]["gamma_effective"] == 0.0
    assert aux["logs"]["residual_ratio"] == 0.0


def test_ssr_fft_identity_variant_reconstructs_without_nan() -> None:
    block = SSRFullBlock(channels=8, num_bands=4, block_variant="s3r_fft_identity")
    x = torch.randn(2, 8, 16, 16)

    y, aux = block(x, return_logs=True)

    assert torch.allclose(y, x, atol=1e-5)
    assert torch.isfinite(y).all()
    assert aux["logs"]["fft_reconstruction_error"] < 1e-5


def test_ssr_block_variants_are_finite() -> None:
    x = torch.randn(2, 8, 16, 16)
    boundary = torch.zeros(2, 1, 16, 16)
    variants = ("s3r_full", "s3r_gamma0", "s3r_fft_identity", "s3r_fixed_band", "s3r_no_suppress", "s3r_simple_spectral")

    for variant in variants:
        block = SSRFullBlock(channels=8, num_bands=4, block_variant=variant)
        y, aux = block(x, boundary_mask=boundary, return_logs=True)
        assert y.shape == x.shape
        assert torch.isfinite(y).all(), variant
        assert torch.isfinite(aux["gate_reg"]).all(), variant
        assert torch.isfinite(aux["hf_ratio_penalty"]).all(), variant
        for value in aux["logs"].values():
            if isinstance(value, list):
                assert all(math.isfinite(float(v)) for v in value), variant
            elif isinstance(value, (float, int)):
                assert math.isfinite(float(value)), variant


def test_spectral_state_transition_and_modulation_shapes() -> None:
    feat = torch.randn(2, 8, 16, 16)
    init = SpectralStateInitializer(channels=8, num_bands=4, state_dim=6)
    state = init(feat)

    transition = SpectralStateTransition(channels=8, num_bands=4, state_dim=6)
    new_state, logs = transition(feat, state)
    mod = StateGuidedModulation(channels=8, num_bands=4, state_dim=6)
    out = mod(feat, new_state)

    assert state.shape == (2, 4, 6)
    assert new_state.shape == state.shape
    assert out.shape == feat.shape
    assert logs["state_norm"].shape == (2, 4)
    assert logs["state_delta"].shape == (2, 4)
    assert logs["state_retain_gate"].shape == (2, 4)
    assert logs["state_update_gate"].shape == (2, 4)


def test_s3r_mini_outputs_segmentation_boundary_and_logs() -> None:
    model = S3RMini(in_channels=1, base_channels=8, num_classes=4, num_bands=4, state_dim=8)
    x = torch.randn(2, 1, 32, 32)
    mask = torch.randint(0, 4, (2, 32, 32))
    boundary = boundary_map_from_mask(mask)

    outputs = model(x, boundary_mask=boundary, return_logs=True)

    assert outputs["seg_logits"].shape == (2, 4, 32, 32)
    assert outputs["boundary_logits"].shape == (2, 1, 32, 32)
    assert outputs["output"].shape == outputs["seg_logits"].shape
    assert outputs["gate_reg"].ndim == 0
    assert outputs["hf_ratio_penalty"].ndim == 0
    assert "state" in outputs
    assert outputs["state"].shape[:2] == (2, 4)
    assert "transitions" in outputs["logs"]


def test_s3r_net_preserves_input_resolution() -> None:
    model = S3RNet(
        in_channels=1,
        base_channels=8,
        stage_channels=(8, 12, 16),
        stage_blocks=(1, 1, 1),
        num_classes=4,
        num_bands=4,
        state_dim=8,
    )
    x = torch.randn(1, 1, 32, 32)

    outputs = model(x, return_logs=True)

    assert outputs["seg_logits"].shape == (1, 4, 32, 32)
    assert outputs["boundary_logits"].shape == (1, 1, 32, 32)
    assert "state" in outputs
    assert "transitions" in outputs["logs"]


def test_s3r_loss_is_finite_for_synthetic_batch() -> None:
    model = S3RMini(in_channels=1, base_channels=8, num_classes=4, num_bands=4, state_dim=8)
    criterion = S3RLoss(num_classes=4, num_bands=4)
    x = torch.randn(2, 1, 32, 32)
    mask = torch.randint(0, 4, (2, 32, 32))
    boundary = boundary_map_from_mask(mask)
    outputs = model(x, boundary_mask=boundary, return_logs=True)

    loss, parts = criterion(outputs, mask)

    assert math.isfinite(float(loss.detach()))
    for key in ("ce", "dice", "boundary_bce", "boundary_dice", "boundary_frequency", "loss"):
        assert key in parts
        assert math.isfinite(parts[key])
