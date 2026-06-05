"""CineMA frozen teacher adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .base_teacher import FrozenSegmentationTeacher, TeacherLoadError
from .teacher_utils import entropy_confidence, list_checkpoint_candidates, make_teacher_stub, normalize_probs, one_hot_to_boundary


class CineMATeacher(FrozenSegmentationTeacher):
    """Cardiac anatomical / boundary teacher wrapper."""

    AUTOSELECT_KEYWORDS = ("seg", "segment", "ventricle", "myocardium", "acdc", "cine", "sax")

    def __init__(
        self,
        checkpoint_dir: str | Path | None,
        device: str | torch.device,
        num_classes: int,
        image_size: int | None = None,
        repo_path: str | Path = "external/CineMA",
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        dataset: str = "acdc",
        view: str = "sax",
        seed: int = 0,
        class_map: str | dict[int, int] | None = None,
        teacher_stub: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_dir, device, num_classes, image_size, teacher_stub, **kwargs)
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.config_path = Path(config_path) if config_path else None
        self.dataset = str(dataset).lower()
        self.view = str(view).lower()
        self.seed = int(seed)
        self.class_map = parse_class_map(class_map)
        self.model = None

    def load(self) -> "CineMATeacher":
        if self.teacher_stub:
            self.loaded = True
            return self
        if not self.repo_path.exists():
            raise TeacherLoadError(f"CineMA repo not found at {self.repo_path}. Run: bash scripts/setup_teachers.sh")
        ckpt = self.checkpoint_path or self.auto_select_checkpoint()
        if ckpt is None:
            raise TeacherLoadError(
                f"No compatible CineMA checkpoint found under {self.checkpoint_dir}. "
                "Prefer the ACDC SAX segmentation checkpoint such as finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors. "
                "If the Hugging Face repo is gated, run: huggingface-cli login"
            )
        self.checkpoint_path = ckpt
        self.config_path = self.config_path or self.auto_select_config(ckpt)
        if self.config_path is None:
            raise TeacherLoadError(
                f"No CineMA config.yaml found near checkpoint {ckpt}. "
                "Expected finetuned/segmentation/acdc_sax/config.yaml or pass --cinema_config."
            )
        sys.path.insert(0, str(self.repo_path.resolve()))
        try:
            cinema_module = importlib.import_module("cinema")
            ConvUNetR = getattr(cinema_module, "ConvUNetR")
        except Exception as exc:  # pragma: no cover - depends on external repo
            raise TeacherLoadError(
                "Could not import `ConvUNetR` from CineMA. Install CineMA first, for example:\n"
                "  cd external/CineMA && pip install -e .\n"
                f"Original import error: {exc}"
            ) from exc

        attempts = []
        try:
            self.model = _load_local_convunetr(self.checkpoint_path, self.config_path)
            self.loaded = True
            self.freeze_model()
            self.meta = {
                "checkpoint_path": str(self.checkpoint_path),
                "config_path": str(self.config_path),
                "dataset": self.dataset,
                "view": self.view,
                "seed": self.seed,
                "backend": "local_safetensors",
            }
            return self
        except Exception as exc:  # pragma: no cover - depends on external repo
            attempts.append(f"local_safetensors {type(exc).__name__}: {exc}")

        rel_model = _relative_or_absolute(self.checkpoint_path, self.checkpoint_dir)
        rel_config = _relative_or_absolute(self.config_path, self.checkpoint_dir)
        for kwargs in (
            {
                "repo_id": "mathpluscode/CineMA",
                "model_filename": rel_model,
                "config_filename": rel_config,
                "cache_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            },
            {
                "repo_id": "mathpluscode/CineMA",
                "model_filename": f"finetuned/segmentation/{self.dataset}_{self.view}/{self.dataset}_{self.view}_{self.seed}.safetensors",
                "config_filename": f"finetuned/segmentation/{self.dataset}_{self.view}/config.yaml",
                "cache_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            },
        ):
            clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                self.model = ConvUNetR.from_finetuned(**clean_kwargs)
                self.loaded = True
                self.freeze_model()
                self.meta = {
                    "checkpoint_path": str(self.checkpoint_path),
                    "config_path": str(self.config_path),
                    "dataset": self.dataset,
                    "view": self.view,
                    "seed": self.seed,
                    "backend": "from_finetuned",
                    "from_finetuned_kwargs": clean_kwargs,
                }
                break
            except Exception as exc:  # pragma: no cover - depends on external repo
                attempts.append(f"{type(exc).__name__}: {exc}")
        else:
            raise TeacherLoadError(
                "CineMA ConvUNetR.from_finetuned failed for local checkpoint/config. "
                f"checkpoint={self.checkpoint_path}, config={self.config_path}. "
                "If this was downloaded with snapshot_download --local_dir, try passing the explicit checkpoint/config paths:\n"
                "  --cinema_ckpt checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors\n"
                "  --cinema_config checkpoints/teachers/cinema/finetuned/segmentation/acdc_sax/config.yaml\n"
                "Attempts: "
                + " | ".join(attempts[:3])
            )
        return self

    def auto_select_checkpoint(self) -> Path | None:
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None
        candidates = list_checkpoint_candidates(self.checkpoint_dir)
        if not candidates:
            return None
        scored = []
        for path in candidates:
            text = str(path).lower()
            score = sum(1 for key in self.AUTOSELECT_KEYWORDS if key in text)
            scored.append((score, len(str(path)), path))
        scored.sort(key=lambda item: (-item[0], item[1], str(item[2])))
        return scored[0][2] if scored[0][0] > 0 else None

    def auto_select_config(self, checkpoint: Path) -> Path | None:
        candidates = []
        if checkpoint.parent.exists():
            candidates.append(checkpoint.parent / "config.yaml")
        if self.checkpoint_dir is not None:
            candidates.extend(sorted(Path(self.checkpoint_dir).glob("**/finetuned/segmentation/*/config.yaml")))
            candidates.extend(sorted(Path(self.checkpoint_dir).glob("**/config.yaml")))
        for path in candidates:
            if path.exists():
                return path
        return None

    @torch.no_grad()
    def predict(self, batch: dict[str, Any]) -> dict[str, Any]:
        if self.teacher_stub:
            return make_teacher_stub(batch, self.num_classes, mode="cinema")
        if self.model is None:
            raise TeacherLoadError("CineMA model is not loaded.")
        image = batch["image"].to(self.device).float()
        original_hw = tuple(image.shape[-2:])
        image = _minmax_scale(image)
        if image.shape[1] > 1:
            sax = image.unsqueeze(1).permute(0, 1, 3, 4, 2).contiguous()  # [B,1,H,W,C]
        else:
            sax = image.unsqueeze(-1)  # [B,1,H,W,1]
        if sax.shape[-1] < 16:
            sax = F.pad(sax, (0, 16 - sax.shape[-1]))
        model_batch = {self.view: sax}
        raw = self.model(model_batch)
        if isinstance(raw, dict):
            logits = raw.get(self.view)
            if logits is None:
                logits = next((value for value in raw.values() if torch.is_tensor(value)), None)
        else:
            logits = raw
        if not torch.is_tensor(logits):
            raise TeacherLoadError(f"CineMA output did not contain tensor logits. Got: {type(raw)!r}")
        if logits.ndim == 5:
            source_depth = int(batch["image"].shape[1])
            center_idx = max(source_depth // 2, 0)
            logits_2d = logits[..., center_idx]
        elif logits.ndim == 4:
            logits_2d = logits
        else:
            raise TeacherLoadError(f"Expected CineMA logits [B,C,H,W,Z] or [B,C,H,W], got {tuple(logits.shape)}")
        if logits_2d.shape[-2:] != original_hw:
            logits_2d = F.interpolate(logits_2d.float(), size=original_hw, mode="bilinear", align_corners=False)
        probs = normalize_probs(torch.softmax(logits_2d.float(), dim=1))
        if self.class_map:
            probs = _remap_probs(probs, self.class_map, self.num_classes)
        confidence = entropy_confidence(probs)
        boundary = one_hot_to_boundary(probs, mode="soft", dilation=3)
        return {
            "logits": logits_2d.detach(),
            "probs": probs.detach(),
            "mask": probs.argmax(dim=1).detach(),
            "confidence": confidence.detach(),
            "boundary": boundary.detach(),
            "meta": {
                "teacher_stub": False,
                "teacher": "cinema",
                "checkpoint_path": str(self.checkpoint_path),
                "config_path": str(self.config_path),
                "view": self.view,
                "class_map": self.class_map,
            },
        }


def parse_class_map(value: str | dict[int, int] | None) -> dict[int, int] | None:
    """Parse `source:target,source:target` class mapping."""
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    mapping: dict[int, int] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        source, target = item.split(":")
        mapping[int(source.strip())] = int(target.strip())
    return mapping


def _relative_or_absolute(path: Path | None, root: Path | None) -> str | None:
    if path is None:
        return None
    if root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _minmax_scale(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(1)
    low = flat.amin(dim=1).view(-1, 1, 1, 1)
    high = flat.amax(dim=1).view(-1, 1, 1, 1)
    return ((x - low) / (high - low).clamp_min(eps)).clamp(0.0, 1.0)


def _remap_probs(probs: torch.Tensor, mapping: dict[int, int], num_classes: int) -> torch.Tensor:
    out = torch.zeros(probs.shape[0], num_classes, *probs.shape[-2:], device=probs.device, dtype=probs.dtype)
    for source, target in mapping.items():
        if 0 <= source < probs.shape[1] and 0 <= target < num_classes:
            out[:, target] += probs[:, source]
    missing = out.sum(dim=1, keepdim=True) <= 0
    if missing.any():
        out = out + missing.float() * probs[:, :1].expand_as(out) / float(num_classes)
    return normalize_probs(out)


def _load_local_convunetr(checkpoint_path: Path, config_path: Path) -> torch.nn.Module:
    """Load CineMA ConvUNetR directly from local safetensors/config files."""
    if not checkpoint_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing local checkpoint/config: {checkpoint_path}, {config_path}")
    try:
        from omegaconf import OmegaConf
        from safetensors import safe_open
        from cinema.segmentation.convunetr import get_model
    except Exception as exc:
        raise TeacherLoadError(
            "Local CineMA loader requires omegaconf, safetensors, and cinema.segmentation.convunetr.get_model. "
            "Install CineMA dependencies with: cd external/CineMA && pip install -e ."
        ) from exc
    config = OmegaConf.load(config_path)
    model = get_model(config)
    state_dict = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=False)
    return model
