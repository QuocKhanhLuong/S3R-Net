"""
SpecMambaNet — Dual-Stream FFT + PseudoMamba in U-Net Encoder-Decoder

Architecture:
- SpectralBlock: channel mixing in Fourier domain via rfft2/irfft2 + Conv1x1
- PseudoMambaBlock: SSM-style causal depthwise Conv1d + gating (pure PyTorch)
- SpecMambaBlock: dual-stream (spectral + mamba) fused with residual
- SpecMambaNet: U-Net encoder-decoder with skip connections + deep supervision

Pure PyTorch — no external mamba_ssm dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBlock(nn.Module):
    """Channel mixing in Fourier domain using real FFT."""

    def __init__(self, dim):
        super().__init__()
        self.conv_real = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv_imag = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            x_f = x.float()
            x_freq = torch.fft.rfft2(x_f, norm='ortho')
            out_real = self.conv_real(x_freq.real)
            out_imag = self.conv_imag(x_freq.imag)
            x_freq_mixed = torch.complex(out_real, out_imag)
            x_out = torch.fft.irfft2(x_freq_mixed, s=(H, W), norm='ortho')
        return self.act(self.norm(x_out))


class PseudoMambaBlock(nn.Module):
    """SSM-style block using causal depthwise Conv1d + gating (pure PyTorch)."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_gate = nn.Linear(dim, dim, bias=False)
        self.dw_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=dim,
            bias=False,
        )
        self.linear_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        seq = x.view(B, C, N).transpose(1, 2)
        h = self.act(self.linear_in(seq))
        g = torch.sigmoid(self.linear_gate(seq))
        h_t = h.transpose(1, 2)
        h_t = self.dw_conv(h_t)[..., :N]
        h = h_t.transpose(1, 2)
        h = h * g
        out = self.linear_out(h)
        x_out = out.transpose(1, 2).view(B, C, H, W)
        return self.act(self.norm(x_out))


class SpecMambaBlock(nn.Module):
    """Dual-stream block: spectral (FFT) + pseudo-Mamba, fused with residual."""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, dim), dim)
        self.norm2 = nn.GroupNorm(min(8, dim), dim)
        self.spectral = SpectralBlock(dim)
        self.mamba = PseudoMambaBlock(dim)
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm_out = nn.GroupNorm(min(8, dim), dim)
        self.act = nn.GELU()

    def forward(self, x):
        x_spec = self.spectral(self.norm1(x))
        x_mamba = self.mamba(self.norm2(x))
        x_fused = self.fuse(torch.cat([x_spec, x_mamba], dim=1))
        return x + self.act(self.norm_out(x_fused))


class SpecMambaNet(nn.Module):
    """U-Net encoder-decoder with SpecMambaBlock at every stage.

    Args:
        in_channels: input image channels (e.g. 3 for RGB, 4 for multi-modal)
        num_classes: number of segmentation classes
        base_channels: channel width at the first encoder stage (C)
        img_size: spatial resolution (used only for compatibility, not hard-coded)
        deep_supervision: if True, return auxiliary outputs from decoder stages
    """

    def __init__(self, in_channels=3, num_classes=4, base_channels=48,
                 img_size=224, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        C = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, C), C),
            nn.GELU(),
        )

        # Encoder
        self.enc1 = SpecMambaBlock(C)
        self.down1 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C * 2), C * 2), nn.GELU(),
        )
        self.enc2 = SpecMambaBlock(C * 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C * 4), C * 4), nn.GELU(),
        )
        self.enc3 = SpecMambaBlock(C * 4)
        self.down3 = nn.Sequential(
            nn.Conv2d(C * 4, C * 8, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, C * 8), C * 8), nn.GELU(),
        )
        self.bottleneck = SpecMambaBlock(C * 8)

        # Decoder
        self.up3_fuse = nn.Conv2d(C * 8 + C * 4, C * 4, 1)
        self.dec3 = SpecMambaBlock(C * 4)

        self.up2_fuse = nn.Conv2d(C * 4 + C * 2, C * 2, 1)
        self.dec2 = SpecMambaBlock(C * 2)

        self.up1_fuse = nn.Conv2d(C * 2 + C, C, 1)
        self.dec1 = SpecMambaBlock(C)

        # Segmentation head
        self.seg_head = nn.Conv2d(C, num_classes, 1)

        # Deep supervision aux heads (from decoder stages at lower resolutions)
        if deep_supervision:
            self.aux_head_3 = nn.Conv2d(C * 4, num_classes, 1)
            self.aux_head_2 = nn.Conv2d(C * 2, num_classes, 1)
            self.aux_head_1 = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        target = x.shape[2:]

        x = self.stem(x)

        feat1 = self.enc1(x)
        feat2 = self.enc2(self.down1(feat1))
        feat3 = self.enc3(self.down2(feat2))
        feat4 = self.bottleneck(self.down3(feat3))

        up3 = F.interpolate(feat4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = self.dec3(self.up3_fuse(torch.cat([up3, feat3], dim=1)))

        up2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = self.dec2(self.up2_fuse(torch.cat([up2, feat2], dim=1)))

        up1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = self.dec1(self.up1_fuse(torch.cat([up1, feat1], dim=1)))

        logits = F.interpolate(self.seg_head(dec1), target, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            aux3 = F.interpolate(self.aux_head_3(dec3), target, mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux_head_2(dec2), target, mode='bilinear', align_corners=True)
            aux1 = F.interpolate(self.aux_head_1(dec1), target, mode='bilinear', align_corners=True)
            return {'output': logits, 'aux_outputs': [aux3, aux2, aux1]}

        return {'output': logits}


# =========================================================================
# Factory functions
# =========================================================================

def specmamba_small(num_classes=4, in_channels=3, deep_supervision=False):
    """Small config: base_channels=32"""
    return SpecMambaNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=32,
        deep_supervision=deep_supervision,
    )


def specmamba_base(num_classes=4, in_channels=3, deep_supervision=False):
    """Base config: base_channels=48"""
    return SpecMambaNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=48,
        deep_supervision=deep_supervision,
    )


def specmamba_large(num_classes=4, in_channels=3, deep_supervision=False):
    """Large config: base_channels=64"""
    return SpecMambaNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=64,
        deep_supervision=deep_supervision,
    )
