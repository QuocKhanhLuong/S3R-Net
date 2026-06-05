#!/usr/bin/env python3
"""Download Medical-SAM3 and CineMA teacher weights from Hugging Face."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINT_EXTENSIONS = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download frozen teacher checkpoints")
    parser.add_argument("--medical_sam3_repo", default="ChongCong/Medical-SAM3")
    parser.add_argument("--cinema_repo", default="mathpluscode/CineMA")
    parser.add_argument("--output_dir", default="checkpoints/teachers")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install with: pip install -U huggingface_hub safetensors") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads = [
        ("medical_sam3", args.medical_sam3_repo, output_dir / "medical_sam3"),
        ("cinema", args.cinema_repo, output_dir / "cinema"),
    ]
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teachers": {},
    }
    for name, repo_id, local_dir in downloads:
        try:
            path = snapshot_download(
                repo_id=repo_id,
                revision=args.revision,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            print(f"Failed to download {name} from {repo_id}: {exc}")
            print("If the repository is gated/private, run: huggingface-cli login")
            raise SystemExit(1) from exc

        candidates = sorted(
            str(p.relative_to(local_dir))
            for p in Path(path).rglob("*")
            if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
        )
        manifest["teachers"][name] = {
            "source_repo": repo_id,
            "local_path": str(local_dir),
            "candidate_checkpoint_files": candidates,
        }
        print(f"{name}: downloaded to {local_dir}; {len(candidates)} checkpoint candidates")
        for item in candidates[:20]:
            print(f"  - {item}")
        if len(candidates) > 20:
            print(f"  ... {len(candidates) - 20} more")

    manifest_path = output_dir / "teacher_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
