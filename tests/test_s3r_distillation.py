from __future__ import annotations

import math
import os
import builtins
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.distillation.distill_losses import S3RSCSDLoss
from src.distillation.teacher_cache import load_teacher_cache, save_teacher_cache, validate_cache
from src.distillation.teacher_targets import (
    boundary_from_mask,
    distance_map_from_mask_or_boundary,
    region_routing_weights,
    semantic_entropy,
    soft_boundary_from_prob,
    spectral_boundary_target,
    teacher_agreement_weight,
)
from src.losses.agreement_kd import agreement_aware_fusion, compute_dual_teacher_kd_loss
from src.teachers import CineMATeacher, MedSAM2Teacher


def test_teacher_cache_round_trip_and_validation(tmp_path: Path) -> None:
    payload = {
        "t1_probs": np.zeros((4, 16, 16), dtype=np.float32),
        "t1_entropy": np.zeros((1, 16, 16), dtype=np.float32),
        "t1_foreground": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_boundary": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_distance": np.zeros((1, 16, 16), dtype=np.float32),
        "t2_foreground": np.zeros((1, 16, 16), dtype=np.float32),
        "agreement_weight": np.ones((1, 16, 16), dtype=np.float32),
    }

    path = save_teacher_cache(tmp_path, "patient001_ED", 3, payload, dataset="ACDC")
    loaded = load_teacher_cache(tmp_path, "patient001_ED", 3, dataset="ACDC")
    report = validate_cache(tmp_path, dataset="ACDC")

    assert path.exists()
    assert loaded["t1_probs"].shape == (4, 16, 16)
    assert report["valid_files"] == 1
    assert report["missing_required_fields"] == {}


def test_teacher_targets_produce_expected_shapes() -> None:
    mask = torch.zeros(2, 16, 16, dtype=torch.long)
    mask[:, 4:10, 4:10] = 1
    probs = torch.nn.functional.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    boundary = boundary_from_mask(mask)
    soft_boundary = soft_boundary_from_prob(probs[:, 1:].sum(dim=1, keepdim=True))
    distance = distance_map_from_mask_or_boundary(mask)
    spectral = spectral_boundary_target(boundary, num_bands=4)
    entropy = semantic_entropy(probs)
    agreement = teacher_agreement_weight(probs[:, 1:].sum(dim=1, keepdim=True), boundary)
    routing = region_routing_weights(
        t1_probs=probs,
        t2_boundary=boundary,
        gt_mask=mask,
        agreement_weight=agreement,
    )

    assert boundary.shape == (2, 1, 16, 16)
    assert soft_boundary.shape == (2, 1, 16, 16)
    assert distance.shape == (2, 1, 16, 16)
    assert spectral.shape == (2, 4)
    assert entropy.shape == (2, 1, 16, 16)
    assert routing["semantic"].shape == (2, 1, 16, 16)
    assert routing["characteristic"].shape == (2, 1, 16, 16)
    assert routing["uncertain"].shape == (2, 1, 16, 16)


def test_s3r_scsd_loss_supports_dual_routing_and_state_kd() -> None:
    student = {
        "seg_logits": torch.randn(2, 4, 16, 16),
        "boundary_logits": torch.randn(2, 1, 16, 16),
        "distance": torch.randn(2, 1, 16, 16),
        "state": torch.randn(2, 4, 8),
    }
    gt = torch.randint(0, 4, (2, 16, 16))
    teacher = {
        "t1_probs": torch.softmax(torch.randn(2, 4, 16, 16), dim=1),
        "t1_entropy": torch.rand(2, 1, 16, 16),
        "t1_foreground": torch.rand(2, 1, 16, 16),
        "t2_boundary": torch.rand(2, 1, 16, 16),
        "t2_distance": torch.rand(2, 1, 16, 16),
        "t2_foreground": torch.rand(2, 1, 16, 16),
        "agreement_weight": torch.rand(2, 1, 16, 16),
    }
    loss_fn = S3RSCSDLoss(
        num_classes=4,
        num_bands=4,
        phase="phase4_state_kd",
        loss_weights={
            "semantic_kd": 0.3,
            "boundary_kd": 0.2,
            "distance_kd": 0.05,
            "spectral_boundary_kd": 0.1,
            "state_kd": 0.05,
            "agreement_weighting": True,
            "region_routing": True,
        },
    )

    loss, parts = loss_fn(student, gt, teacher)

    assert math.isfinite(float(loss.detach()))
    for key in ("semantic_kd", "boundary_kd", "distance_kd", "spectral_boundary_kd", "state_kd"):
        assert key in parts
        assert math.isfinite(parts[key])


def test_agreement_weighting_applies_without_region_routing() -> None:
    student = {
        "seg_logits": torch.randn(1, 4, 12, 12),
        "boundary_logits": torch.randn(1, 1, 12, 12),
    }
    teacher = {
        "t1_probs": torch.softmax(torch.randn(1, 4, 12, 12), dim=1),
        "t2_boundary": torch.rand(1, 1, 12, 12),
        "agreement_weight": torch.zeros(1, 1, 12, 12),
    }
    loss_fn = S3RSCSDLoss(
        num_classes=4,
        loss_weights={"semantic_kd": 1.0, "boundary_kd": 1.0, "agreement_weighting": True, "region_routing": False},
    )

    loss, parts = loss_fn(student, None, teacher)

    assert math.isfinite(float(loss.detach()))
    assert parts["agreement_mean"] == 0.0
    assert parts["semantic_kd"] == 0.0
    assert parts["boundary_kd"] == 0.0


def test_agreement_aware_fusion_normalizes_and_bounds_weights() -> None:
    B, C, H, W = 2, 4, 16, 16
    p_m3 = torch.softmax(torch.randn(B, C, H, W), dim=1)
    p_c = torch.softmax(torch.randn(B, C, H, W), dim=1)
    fusion = agreement_aware_fusion(
        p_m3,
        p_c,
        torch.rand(B, 1, H, W),
        torch.rand(B, 1, H, W),
        gt_mask=torch.randint(0, C, (B, H, W)),
    )

    assert fusion["P_F"].shape == (B, C, H, W)
    assert torch.allclose(fusion["P_F"].sum(dim=1), torch.ones(B, H, W), atol=1e-5)
    for key in ("agreement", "W_M3", "W_C", "W_boundary", "W_interior"):
        assert torch.isfinite(fusion[key]).all()
        assert float(fusion[key].min()) >= 0.0
        assert float(fusion[key].max()) <= 1.0


def test_dual_teacher_kd_loss_and_stub_outputs_are_finite() -> None:
    batch = {
        "image": torch.randn(2, 1, 16, 16),
        "mask": torch.randint(0, 4, (2, 16, 16)),
    }
    m3 = MedSAM2Teacher(None, device="cpu", num_classes=4, teacher_stub=True)
    cinema = CineMATeacher(None, device="cpu", num_classes=4, teacher_stub=True)
    out_m3 = m3(batch)
    out_c = cinema(batch)
    teacher = {
        "P_M3": out_m3["probs"],
        "C_M3": out_m3["confidence"],
        "P_C": out_c["probs"],
        "C_C": out_c["confidence"],
        "B_C": out_c["boundary"],
    }
    student = {"seg_logits": torch.randn(2, 4, 16, 16)}

    loss, parts, fusion = compute_dual_teacher_kd_loss(student, teacher, batch["mask"], {"dual_teacher_kd": {}})

    assert math.isfinite(float(loss.detach()))
    assert parts["loss_kd"] >= 0.0
    assert fusion["P_F"].shape == (2, 4, 16, 16)


def test_medsam2_real_adapter_binds_video_predictor(tmp_path: Path) -> None:
    repo = tmp_path / "MedSAM2"
    sam2_dir = repo / "sam2"
    sam2_dir.mkdir(parents=True)
    (sam2_dir / "__init__.py").write_text("", encoding="utf-8")
    cfg = sam2_dir / "configs" / "sam2.1_hiera_t512.yaml"
    cfg.parent.mkdir()
    cfg.write_text("model: {}\n", encoding="utf-8")
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "older.pt").write_bytes(b"old")
    (ckpt_dir / "MedSAM2_latest.pt").write_bytes(b"fake")
    (sam2_dir / "build_sam.py").write_text(
        """
import numpy as np
import torch

class FakePredictor:
    def __init__(self, cfg, checkpoint):
        self.cfg = cfg
        self.checkpoint = checkpoint

    def init_state(self, image, video_height, video_width):
        return {'height': video_height, 'width': video_width}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box=None, **kwargs):
        h = inference_state['height']
        w = inference_state['width']
        x_min, y_min, x_max, y_max = [int(v) for v in box]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[max(y_min, 0):min(y_max + 1, h), max(x_min, 0):min(x_max + 1, w)] = 1
        logits = torch.from_numpy(mask).float().view(1, 1, h, w)
        return None, [obj_id], logits

    def propagate_in_video(self, inference_state, reverse=False):
        return []

    def reset_state(self, inference_state):
        pass

def build_sam2_video_predictor_npz(cfg, checkpoint):
    assert cfg == 'configs/sam2.1_hiera_t512.yaml'
    return FakePredictor(cfg, checkpoint)
""",
        encoding="utf-8",
    )
    batch = {
        "image": torch.rand(1, 5, 16, 16),
        "mask": torch.zeros(1, 16, 16, dtype=torch.long),
    }
    batch["mask"][:, 4:10, 5:12] = 1
    teacher = MedSAM2Teacher(
        ckpt_dir,
        device="cpu",
        num_classes=4,
        image_size=16,
        repo_path=repo,
        config_path=cfg,
        teacher_stub=False,
    )

    out = teacher(batch)

    assert teacher.checkpoint_path is not None
    assert teacher.checkpoint_path.name == "MedSAM2_latest.pt"
    assert teacher.config_name == "configs/sam2.1_hiera_t512.yaml"
    assert out["probs"].shape == (1, 4, 16, 16)
    assert out["confidence"].shape == (1, 1, 16, 16)
    assert out["boundary"].shape == (1, 1, 16, 16)
    assert torch.isfinite(out["probs"]).all()
    assert float(out["probs"][:, 1].max()) > 0.0
    assert out["meta"]["teacher"] == "medsam2"


def test_medsam2_checkpoint_resolver_accepts_project_relative_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints" / "teachers" / "medsam2"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "MedSAM2_latest.pt"
    ckpt.write_bytes(b"fake")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        teacher = MedSAM2Teacher(
            "checkpoints/teachers/medsam2",
            device="cpu",
            num_classes=4,
            checkpoint_path="checkpoints/teachers/medsam2/MedSAM2_latest.pt",
        )

        assert teacher._resolve_checkpoint() == ckpt.resolve()
    finally:
        os.chdir(cwd)


def test_medsam2_config_resolver_returns_hydra_config_name(tmp_path: Path) -> None:
    repo = tmp_path / "MedSAM2"
    cfg = repo / "sam2" / "configs" / "sam2.1_hiera_t512.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("model: {}\n", encoding="utf-8")
    teacher = MedSAM2Teacher(
        tmp_path,
        device="cpu",
        num_classes=4,
        repo_path=repo,
        config_path=cfg,
    )

    assert teacher._resolve_config_name() == "configs/sam2.1_hiera_t512.yaml"
    legacy = MedSAM2Teacher(
        tmp_path,
        device="cpu",
        num_classes=4,
        repo_path=repo,
        config_path=repo / "configs" / "sam2.1_hiera_t512.yaml",
    )
    assert legacy._resolve_config_name() == "configs/sam2.1_hiera_t512.yaml"


def test_teacher_loading_preview_skips_when_matplotlib_missing(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("test_teacher_loading_script", "scripts/test_teacher_loading.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = {
        "image": torch.randn(1, 1, 16, 16),
        "mask": torch.zeros(1, 16, 16, dtype=torch.long),
    }
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("missing matplotlib")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        saved = module.save_preview(batch, None, None, None, tmp_path / "preview.png")
    finally:
        builtins.__import__ = original_import

    assert saved is False
    assert not (tmp_path / "preview.png").exists()


def test_teacher_scripts_resolve_amp_dtype() -> None:
    for script_path, module_name in (
        ("scripts/test_teacher_loading.py", "test_teacher_loading_script_amp"),
        ("scripts/precompute_teacher_outputs.py", "precompute_teacher_outputs_script_amp"),
    ):
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module._resolve_amp_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16
        assert module._resolve_amp_dtype("float16", torch.device("cpu")) is torch.float16
        try:
            module._resolve_amp_dtype("float32", torch.device("cpu"))
        except ValueError as exc:
            assert "Unsupported teacher AMP dtype" in str(exc)
        else:
            raise AssertionError("unsupported teacher AMP dtype should fail")


def test_real_dual_teacher_config_uses_real_teacher_cache() -> None:
    spec = importlib.util.spec_from_file_location("train_s3r_acdc_real_config", "src/training/train_s3r_acdc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.apply_config_defaults(module.load_config("configs/s3r_dual_teacher_real_100ep.yaml"))
    kd = cfg["dual_teacher_kd"]

    assert cfg["epochs"] == 100
    assert cfg["batch_size"] == 4
    assert cfg["input_mode"] == "25d"
    assert cfg["in_channels"] == 5
    assert cfg["wandb"]["enabled"] is True
    assert cfg["wandb"]["project"] == "s3r-acdc"
    assert cfg["wandb"]["run_name"] == "s3r_dual_teacher_real_100ep"
    assert kd["enabled"] is True
    assert kd["teacher_stub"] is False
    assert kd["medsam2_stub"] is False
    assert kd["cinema_stub"] is False
    assert kd["teacher_cache_dir"] == "teacher_cache/acdc_real"
    assert kd["strict_teacher_cache"] is True
    assert kd["medsam2_repo_path"] == "external/MedSAM2"
    assert kd["medsam2_ckpt"] == "MedSAM2_latest.pt"
    assert kd["cinema_ckpt"].endswith("acdc_sax_0.safetensors")
    assert kd["teacher_amp"] is True
    assert kd["teacher_amp_dtype"] == "bfloat16"
    assert kd["fused_kd_weight_mode"] == "agreement"


def test_train_config_defaults_teacher_amp_dtype() -> None:
    spec = importlib.util.spec_from_file_location("train_s3r_acdc_amp_dtype", "src/training/train_s3r_acdc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.apply_config_defaults({"dual_teacher_kd": {"enabled": True}})

    assert cfg["dual_teacher_kd"]["teacher_amp_dtype"] == "bfloat16"
    assert module._resolve_teacher_amp_dtype("float16", torch.device("cpu")) is torch.float16


def test_pre_epoch_config_table_reports_cached_dual_teacher_and_profile() -> None:
    spec = importlib.util.spec_from_file_location("train_s3r_acdc_startup_table", "src/training/train_s3r_acdc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.apply_config_defaults(module.load_config("configs/s3r_dual_teacher_real_100ep.yaml"))
    kd = cfg["dual_teacher_kd"]
    table = module.format_pre_epoch_config_table(
        cfg,
        torch.device("cuda"),
        {"train_slices": 16, "val_slices": 4},
        {"params": 1_234_567, "flops": 9_876_543_210, "profile_backend": "unit"},
        {
            "cfg": kd,
            "teacher_device": torch.device("cuda"),
            "cache_dir": Path("teacher_cache/acdc_real"),
            "medsam2": None,
            "cinema": None,
        },
    )

    assert "Pre-epoch configuration" in table
    assert "Params" in table and "1.23M" in table
    assert "GFLOPs" in table and "9.88" in table
    assert "SAM2/MedSAM2 field" in table
    assert "enabled via cache" in table
    assert "P_M3/C_M3" in table
    assert "CineMA boundary/anatomy" in table
    assert "P_C/C_C/B_C" in table
    assert "Agreement weighting" in table and "enabled" in table
    assert "Fused KD gate" in table and "agreement" in table


def test_agreement_gated_fused_kd_reports_bounded_weight() -> None:
    gt_mask = torch.zeros(1, 8, 8, dtype=torch.long)
    p_m3 = torch.zeros(1, 4, 8, 8)
    p_m3[:, 1] = 1.0
    p_c = torch.zeros(1, 4, 8, 8)
    p_c[:, 2] = 1.0
    teacher = {
        "P_M3": p_m3,
        "C_M3": torch.ones(1, 1, 8, 8),
        "P_C": p_c,
        "C_C": torch.ones(1, 1, 8, 8),
        "B_C": torch.zeros(1, 1, 8, 8),
    }
    student = {"seg_logits": torch.randn(1, 4, 8, 8)}

    _, parts, fusion = compute_dual_teacher_kd_loss(
        student,
        teacher,
        gt_mask,
        {
            "dual_teacher_kd": {
                "fused_kd_weight_mode": "agreement",
                "fused_kd_min_weight": 0.25,
                "fused_kd_agreement_power": 2.0,
            }
        },
    )

    fuse_weight = fusion["W_fuse"]
    assert torch.isfinite(fuse_weight).all()
    assert float(fuse_weight.min()) >= 0.25
    assert float(fuse_weight.max()) <= 1.0
    assert 0.25 <= parts["fuse_weight_mean"] <= 1.0
    assert 0.25 <= parts["fuse_weight_min"] <= parts["fuse_weight_max"] <= 1.0
    assert 0.0 <= parts["teacher_disagreement_mean"] <= 1.0


def test_agreement_gated_fused_kd_reduces_fuse_loss_against_unweighted_mode() -> None:
    gt_mask = torch.zeros(1, 8, 8, dtype=torch.long)
    student = {"seg_logits": torch.randn(1, 4, 8, 8)}
    p_m3 = torch.zeros(1, 4, 8, 8)
    p_m3[:, 1] = 1.0
    p_c = torch.zeros(1, 4, 8, 8)
    p_c[:, 2] = 1.0
    teacher = {
        "P_M3": p_m3,
        "C_M3": torch.ones(1, 1, 8, 8),
        "P_C": p_c,
        "C_C": torch.ones(1, 1, 8, 8),
        "B_C": torch.zeros(1, 1, 8, 8),
    }

    cfg = {
        "dual_teacher_kd": {
            "lambda_field": 0.0,
            "lambda_cine_boundary": 0.0,
            "lambda_fuse": 1.0,
            "lambda_spec": 0.0,
            "disable_field_kd": True,
            "disable_cine_boundary_kd": True,
            "disable_spectral_kd": True,
            "fused_kd_weight_mode": "none",
        }
    }
    gated_cfg = {
        "dual_teacher_kd": {
            **cfg["dual_teacher_kd"],
            "fused_kd_weight_mode": "agreement",
            "fused_kd_min_weight": 0.05,
            "fused_kd_agreement_power": 1.0,
        }
    }

    _, unweighted_parts, _ = compute_dual_teacher_kd_loss(student, teacher, gt_mask, cfg)
    _, gated_parts, _ = compute_dual_teacher_kd_loss(student, teacher, gt_mask, gated_cfg)

    assert gated_parts["teacher_disagreement_mean"] > 0.0
    assert gated_parts["fuse_weight_mean"] < unweighted_parts["fuse_weight_mean"]
    assert gated_parts["loss_fuse"] < unweighted_parts["loss_fuse"]
