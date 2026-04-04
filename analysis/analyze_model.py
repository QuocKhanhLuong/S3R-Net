
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
import torch.nn as nn
import argparse

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=None, tablefmt=None):
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 60)
        for row in data:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


def analyze_specmamba():
    """Analyze SpecMambaNet architecture."""
    from models.specmamba_net import SpecMambaNet

    parser = argparse.ArgumentParser(description="Analyze Model Structure")
    parser.add_argument('--model', type=str, default='specmamba',
                        choices=['specmamba', 'hrnet_dcn'])
    parser.add_argument('--base_channels', type=int, default=48)
    parser.add_argument('--deep_supervision', action='store_true')
    parser.add_argument('--use_pointrend', action='store_true')
    parser.add_argument('--no_full_res', action='store_true')
    parser.add_argument('--use_shearlet', action='store_true')
    args = parser.parse_args()

    if args.model == 'hrnet_dcn':
        from models.hrnet_dcn import HRNetDCN
        model = HRNetDCN(
            in_channels=3, num_classes=4,
            base_channels=args.base_channels,
            use_pointrend=args.use_pointrend,
            full_resolution_mode=not args.no_full_res,
            deep_supervision=args.deep_supervision,
            use_shearlet=args.use_shearlet,
        )
        arch_name = "HRNetDCN (deprecated)"
    else:
        model = SpecMambaNet(
            in_channels=3, num_classes=4,
            base_channels=args.base_channels,
            deep_supervision=args.deep_supervision,
        )
        arch_name = "SpecMambaNet"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"ANALYZING {arch_name}")
    print(f"{'='*60}")

    print(f"\n[1] PARAMETER COUNT")
    print(f"   - Total Params:     {total_params:,}")
    print(f"   - Trainable Params: {trainable_params:,}")
    print(f"   - Model Size (MB):  {total_params * 4 / 1024 / 1024:.2f} MB")

    if args.model == 'specmamba':
        C = args.base_channels
        print(f"\n[2] ARCHITECTURE DETAILS (Base Channels = {C})")
        stages = [
            ["Stem", f"3 -> {C}", "224x224 (1x)", "Conv3x3 + GN + GELU"],
            ["Encoder 1", f"{C}", "224x224 (1x)", "SpecMambaBlock"],
            ["Encoder 2", f"{C*2}", "112x112 (1/2)", "SpecMambaBlock"],
            ["Encoder 3", f"{C*4}", "56x56 (1/4)", "SpecMambaBlock"],
            ["Bottleneck", f"{C*8}", "28x28 (1/8)", "SpecMambaBlock"],
            ["Decoder 3", f"{C*4}", "56x56 (1/4)", "SpecMambaBlock"],
            ["Decoder 2", f"{C*2}", "112x112 (1/2)", "SpecMambaBlock"],
            ["Decoder 1", f"{C}", "224x224 (1x)", "SpecMambaBlock"],
            ["Head", f"{C} -> 4", "224x224 (1x)", "Conv1x1"],
        ]
        print(tabulate(stages, headers=["Stage", "Channels", "Resolution", "Block"], tablefmt="grid"))

        print(f"\n[3] DETAILED PARAMETERS BY SUBMODULE")
        submodules = [
            ("Stem", model.stem),
            ("Encoder 1", model.enc1), ("Down 1", model.down1),
            ("Encoder 2", model.enc2), ("Down 2", model.down2),
            ("Encoder 3", model.enc3), ("Down 3", model.down3),
            ("Bottleneck", model.bottleneck),
            ("Decoder 3", model.dec3), ("Up3 Fuse", model.up3_fuse),
            ("Decoder 2", model.dec2), ("Up2 Fuse", model.up2_fuse),
            ("Decoder 1", model.dec1), ("Up1 Fuse", model.up1_fuse),
            ("Seg Head", model.seg_head),
        ]
        if args.deep_supervision:
            submodules.append(("Aux Head 3", model.aux_head_3))
            submodules.append(("Aux Head 2", model.aux_head_2))
            submodules.append(("Aux Head 1", model.aux_head_1))

        sub_data = []
        for name, module in submodules:
            params = sum(p.numel() for p in module.parameters())
            sub_data.append([name, f"{params:,}", f"{params/total_params*100:.1f}%"])
        print(tabulate(sub_data, headers=["Module", "Params", "% of Total"], tablefmt="grid"))

    # Smoke test
    print(f"\n[4] FORWARD PASS SMOKE TEST")
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"   Input:  {dummy.shape}")
    print(f"   Output: {out['output'].shape}")
    print(f"   OK!")
    print()


if __name__ == '__main__':
    try:
        analyze_specmamba()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
