"""S3R-Mini and S3R-Net segmentation models."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .s3r_blocks import LargeKernelRefine, SSRFullBlock
from .s3r_state import (
    SpectralStateInitializer,
    SpectralStateTransition,
    StateGuidedModulation,
    detach_state_logs,
)
from .spectral_utils import num_groups


class ConvBlock(nn.Module):
    """Two-convolution block that preserves spatial shape."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SignalLift(nn.Module):
    """Initial image-to-feature lift."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AntiAliasedDownsample(nn.Module):
    """Depthwise blur followed by stride-2 projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.blur = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        with torch.no_grad():
            kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
            self.blur.weight.copy_(kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.blur.weight.requires_grad_(False)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.blur(x))


class ReconstructionBlock(nn.Module):
    """Upsample, project, normalize, and refine geometry."""

    def __init__(self, in_channels: int, out_channels: int, large_kernel_size: int = 7) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(out_channels), out_channels),
            nn.GELU(),
        )
        self.refine = LargeKernelRefine(out_channels, kernel_size=large_kernel_size)

    def forward(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.refine(self.project(x))


class S3RTransitionBlock(nn.Module):
    """One S3R transition `(F_t, S_t) -> (F_{t+1}, S_{t+1})`."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        state_dim: int | None = None,
        ssr: dict[str, Any] | None = None,
        state_modulation_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or channels)
        ssr_cfg = dict(ssr or {})
        ssr_cfg.setdefault("num_bands", self.num_bands)
        self.state_transition = SpectralStateTransition(channels, self.num_bands, self.state_dim)
        self.ssr_full_block = SSRFullBlock(channels, **ssr_cfg)
        self.state_guided_modulation = StateGuidedModulation(
            channels,
            self.num_bands,
            self.state_dim,
            state_modulation_scale=state_modulation_scale,
        )

    def forward(
        self,
        feat: Tensor,
        state: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        state_new, state_logs = self.state_transition(feat, state)
        feat_new, ssr_aux = self.ssr_full_block(feat, boundary_mask=boundary_mask, return_logs=return_logs)
        feat_new = self.state_guided_modulation(feat_new, state_new)

        logs: dict[str, Any] = {}
        if return_logs:
            logs = {
                "ssr": ssr_aux.get("logs", {}),
                "state": detach_state_logs(state_logs),
            }
        return feat_new, state_new, {
            "gate_reg": ssr_aux["gate_reg"],
            "hf_ratio_penalty": ssr_aux["hf_ratio_penalty"],
            "logs": logs,
            "state_logs_raw": state_logs,
        }


class S3RMini(nn.Module):
    """Compact S3R model replacing the MiniSSR prototype."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 4,
        num_bands: int = 4,
        state_dim: int | None = None,
        ssr: dict[str, Any] | None = None,
        use_s3r_state: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or base_channels)
        self.use_s3r_state = bool(use_s3r_state)

        self.stem = SignalLift(in_channels, base_channels)
        self.state_initializer = SpectralStateInitializer(base_channels, num_bands, self.state_dim)
        self.s3r1 = S3RTransitionBlock(base_channels, num_bands, self.state_dim, ssr=ssr)
        self.mid = ConvBlock(base_channels, base_channels)
        self.s3r2 = S3RTransitionBlock(base_channels, num_bands, self.state_dim, ssr=ssr)
        self.final_modulation = StateGuidedModulation(base_channels, num_bands, self.state_dim)
        self.shared_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(base_channels), base_channels),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> dict[str, Any]:
        feat = self.stem(x)
        state = self.state_initializer(feat)
        gate_reg = torch.zeros((), device=x.device, dtype=feat.dtype)
        hf_ratio_penalty = torch.zeros((), device=x.device, dtype=feat.dtype)
        transition_logs: dict[str, Any] = {}
        state_raw_logs: dict[str, Any] = {}

        feat, state, aux = self.s3r1(feat, state, boundary_mask=boundary_mask, return_logs=return_logs)
        gate_reg = gate_reg + aux["gate_reg"]
        hf_ratio_penalty = hf_ratio_penalty + aux["hf_ratio_penalty"]
        if return_logs:
            transition_logs["s3r1"] = aux["logs"]
            state_raw_logs["s3r1"] = aux["state_logs_raw"]

        feat = self.mid(feat)
        feat, state, aux = self.s3r2(feat, state, boundary_mask=boundary_mask, return_logs=return_logs)
        gate_reg = gate_reg + aux["gate_reg"]
        hf_ratio_penalty = hf_ratio_penalty + aux["hf_ratio_penalty"]
        if return_logs:
            transition_logs["s3r2"] = aux["logs"]
            state_raw_logs["s3r2"] = aux["state_logs_raw"]

        feat = self.final_modulation(feat, state)
        feat = self.shared_head(feat)
        seg_logits = self.seg_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs: dict[str, Any] = {
            "seg_logits": seg_logits,
            "output": seg_logits,
            "boundary_logits": boundary_logits,
            "gate_reg": gate_reg,
            "hf_ratio_penalty": hf_ratio_penalty,
            "state": state,
        }
        if return_logs:
            outputs["logs"] = {"transitions": transition_logs, "state_raw": state_raw_logs}
        return outputs


class S3RNet(nn.Module):
    """Future S3R-Net architecture with state-guided reconstruction."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        stage_channels: Iterable[int] = (32, 64, 128),
        stage_blocks: Iterable[int] = (2, 2, 3),
        num_classes: int = 4,
        num_bands: int = 4,
        state_dim: int | None = None,
        ssr: dict[str, Any] | None = None,
        large_kernel_size: int = 7,
        use_state_bridge: bool = False,
    ) -> None:
        super().__init__()
        channels = tuple(int(c) for c in stage_channels)
        blocks = tuple(int(b) for b in stage_blocks)
        if len(channels) != 3 or len(blocks) != 3:
            raise ValueError("stage_channels and stage_blocks must each have length 3")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.stage_channels = channels
        self.stage_blocks = blocks
        self.num_classes = int(num_classes)
        self.num_bands = int(num_bands)
        self.state_dim = int(state_dim or channels[0])
        self.use_state_bridge = bool(use_state_bridge)

        self.lift = SignalLift(in_channels, channels[0])
        self.state_initializer = SpectralStateInitializer(channels[0], num_bands, self.state_dim)
        self.stage1 = nn.ModuleList([S3RTransitionBlock(channels[0], num_bands, self.state_dim, ssr=ssr) for _ in range(blocks[0])])
        self.down1 = AntiAliasedDownsample(channels[0], channels[1])
        self.stage2 = nn.ModuleList([S3RTransitionBlock(channels[1], num_bands, self.state_dim, ssr=ssr) for _ in range(blocks[1])])
        self.down2 = AntiAliasedDownsample(channels[1], channels[2])
        self.stage3 = nn.ModuleList([S3RTransitionBlock(channels[2], num_bands, self.state_dim, ssr=ssr) for _ in range(blocks[2])])
        self.recon2 = ReconstructionBlock(channels[2], channels[1], large_kernel_size)
        self.recon1 = ReconstructionBlock(channels[1], channels[0], large_kernel_size)
        self.final_modulation = StateGuidedModulation(channels[0], num_bands, self.state_dim)
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        boundary_mask: Tensor | None = None,
        return_logs: bool = False,
    ) -> dict[str, Any]:
        size1 = tuple(x.shape[-2:])
        feat = self.lift(x)
        state = self.state_initializer(feat)
        gate_reg = torch.zeros((), device=x.device, dtype=feat.dtype)
        hf_ratio_penalty = torch.zeros((), device=x.device, dtype=feat.dtype)
        transition_logs: dict[str, Any] = {}
        state_raw_logs: dict[str, Any] = {}

        feat, state = self._run_stage("stage1", self.stage1, feat, state, boundary_mask, return_logs, transition_logs, state_raw_logs)
        size_stage1 = tuple(feat.shape[-2:])
        feat = self.down1(feat)
        feat, state = self._run_stage("stage2", self.stage2, feat, state, boundary_mask, return_logs, transition_logs, state_raw_logs)
        size_stage2 = tuple(feat.shape[-2:])
        feat = self.down2(feat)
        feat, state = self._run_stage("stage3", self.stage3, feat, state, boundary_mask, return_logs, transition_logs, state_raw_logs)

        for logs in transition_logs.values():
            pass
        gate_reg, hf_ratio_penalty = self._sum_transition_regs(transition_logs, x.device, feat.dtype)
        feat = self.recon2(feat, size_stage2)
        feat = self.recon1(feat, size_stage1)
        if tuple(feat.shape[-2:]) != size1:
            feat = F.interpolate(feat, size=size1, mode="bilinear", align_corners=False)
        feat = self.final_modulation(feat, state)
        feat = self.head(feat)
        seg_logits = self.seg_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs: dict[str, Any] = {
            "seg_logits": seg_logits,
            "output": seg_logits,
            "boundary_logits": boundary_logits,
            "gate_reg": gate_reg,
            "hf_ratio_penalty": hf_ratio_penalty,
            "state": state,
        }
        if return_logs:
            outputs["logs"] = {"transitions": transition_logs, "state_raw": state_raw_logs}
        return outputs

    def _run_stage(
        self,
        name: str,
        blocks: nn.ModuleList,
        feat: Tensor,
        state: Tensor,
        boundary_mask: Tensor | None,
        return_logs: bool,
        transition_logs: dict[str, Any],
        state_raw_logs: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        for idx, block in enumerate(blocks):
            feat, state, aux = block(feat, state, boundary_mask=boundary_mask, return_logs=return_logs)
            key = f"{name}_{idx + 1}"
            transition_logs[key] = {
                "gate_reg_tensor": aux["gate_reg"],
                "hf_ratio_penalty_tensor": aux["hf_ratio_penalty"],
                **(aux["logs"] if return_logs else {}),
            }
            if return_logs:
                state_raw_logs[key] = aux["state_logs_raw"]
        return feat, state

    @staticmethod
    def _sum_transition_regs(logs: dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        gate = torch.zeros((), device=device, dtype=dtype)
        hf = torch.zeros((), device=device, dtype=dtype)
        for item in logs.values():
            gate = gate + item.get("gate_reg_tensor", torch.zeros((), device=device, dtype=dtype))
            hf = hf + item.get("hf_ratio_penalty_tensor", torch.zeros((), device=device, dtype=dtype))
            item.pop("gate_reg_tensor", None)
            item.pop("hf_ratio_penalty_tensor", None)
        return gate, hf


def build_s3r_model(
    model: str = "s3r_mini",
    in_channels: int = 1,
    base_channels: int = 32,
    num_classes: int = 4,
    num_bands: int = 4,
    state_dim: int | None = None,
    ssr: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build an S3R model by public name."""
    name = str(model).lower()
    if name in {"s3r", "s3r_mini", "mini"}:
        return S3RMini(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            num_bands=num_bands,
            state_dim=state_dim,
            ssr=ssr,
            use_s3r_state=bool(kwargs.get("use_s3r_state", True)),
        )
    if name in {"s3r_net", "s3r_full"}:
        stage_channels = kwargs.get("stage_channels") or (base_channels, base_channels * 2, base_channels * 4)
        stage_blocks = kwargs.get("stage_blocks") or (2, 2, 3)
        return S3RNet(
            in_channels=in_channels,
            base_channels=base_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            num_classes=num_classes,
            num_bands=num_bands,
            state_dim=state_dim,
            ssr=ssr,
        )
    raise ValueError("model must be one of: s3r, s3r_mini, s3r_net")
