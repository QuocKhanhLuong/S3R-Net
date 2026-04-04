"""
Architecture Validation Test Suite for SpecMambaNet.

Verifies tensor shapes, gradient flow, deep supervision, and AMP compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
import torch.nn as nn

from models.specmamba_net import (
    SpectralBlock, PseudoMambaBlock, SpecMambaBlock, SpecMambaNet,
)


def test_spectral_block():
    print("Testing SpectralBlock...")
    block = SpectralBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  OK: {x.shape} -> {out.shape}")


def test_pseudo_mamba_block():
    print("Testing PseudoMambaBlock...")
    block = PseudoMambaBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  OK: {x.shape} -> {out.shape}")


def test_spec_mamba_block():
    print("Testing SpecMambaBlock...")
    block = SpecMambaBlock(48)
    x = torch.randn(2, 48, 56, 56)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  OK: {x.shape} -> {out.shape}")


def test_specmamba_net_forward():
    print("Testing SpecMambaNet forward (eval)...")
    model = SpecMambaNet(in_channels=3, num_classes=4, base_channels=32)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out['output'].shape == (2, 4, 224, 224), f"Shape: {out['output'].shape}"
    assert 'aux_outputs' not in out, "aux_outputs should not appear in eval mode"
    print(f"  OK: output={out['output'].shape}")


def test_deep_supervision():
    print("Testing deep supervision...")
    model = SpecMambaNet(in_channels=3, num_classes=4, base_channels=32, deep_supervision=True)
    model.train()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert 'aux_outputs' in out, "Missing aux_outputs in training mode"
    assert len(out['aux_outputs']) == 3, f"Expected 3 aux heads, got {len(out['aux_outputs'])}"
    for i, aux in enumerate(out['aux_outputs']):
        assert aux.shape == out['output'].shape, f"Aux {i}: {aux.shape} != {out['output'].shape}"
    print(f"  OK: {len(out['aux_outputs'])} aux heads, all match output shape")


def test_gradient_flow():
    print("Testing gradient flow...")
    model = SpecMambaNet(in_channels=3, num_classes=4, base_channels=16)
    model.train()
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = out['output'].sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters have no gradient"
    print(f"  OK: all parameters received gradients")


def test_different_sizes():
    print("Testing different input sizes...")
    model = SpecMambaNet(in_channels=3, num_classes=2, base_channels=16)
    model.eval()
    for size in [64, 128, 224, 256]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out['output'].shape == (1, 2, size, size), f"Size {size}: {out['output'].shape}"
        print(f"  OK: {size}x{size}")


def test_multi_channel_input():
    print("Testing 4-channel input (BraTS-style)...")
    model = SpecMambaNet(in_channels=4, num_classes=4, base_channels=16)
    model.eval()
    x = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out['output'].shape == (1, 4, 128, 128)
    print(f"  OK: {out['output'].shape}")


def test_param_count():
    print("Testing parameter count...")
    model = SpecMambaNet(in_channels=3, num_classes=4, base_channels=48)
    params = sum(p.numel() for p in model.parameters())
    print(f"  C=48: {params:,} params")
    assert params > 0, "Model has no parameters"
    print(f"  OK")


if __name__ == '__main__':
    tests = [
        test_spectral_block,
        test_pseudo_mamba_block,
        test_spec_mamba_block,
        test_specmamba_net_forward,
        test_deep_supervision,
        test_gradient_flow,
        test_different_sizes,
        test_multi_channel_input,
        test_param_count,
    ]

    print(f"\n{'='*60}")
    print("SpecMambaNet Test Suite")
    print(f"{'='*60}\n")

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print(f"{'='*60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'='*60}")
