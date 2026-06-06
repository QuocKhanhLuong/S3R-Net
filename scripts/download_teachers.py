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
    parser.add_argument("--teacher", choices=["medical_sam3", "cinema", "both"], default="both")
    parser.add_argument("--medical_sam3_repo", default="ChongCong/Medical-SAM3")
    parser.add_argument("--cinema_repo", default="mathpluscode/CineMA")
    parser.add_argument("--output_dir", default="checkpoints/teachers")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow_patterns", default=None, help="Comma-separated HF allow_patterns override.")
    parser.add_argument("--ignore_patterns", default=None, help="Comma-separated HF ignore_patterns override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install with: pip install -U huggingface_hub safetensors") from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    downloads = []
    if args.teacher in {"medical_sam3", "both"}:
        downloads.append(("medical_sam3", args.medical_sam3_repo, output_dir / "medical_sam3", ["checkpoint.pt"]))
    if args.teacher in {"cinema", "both"}:
        downloads.append(("cinema", args.cinema_repo, output_dir / "cinema", None))
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teachers": {},
    }
    allow_override = _split_patterns(args.allow_patterns)
    ignore_override = _split_patterns(args.ignore_patterns)
    for name, repo_id, local_dir, default_allow_patterns in downloads:
        allow_patterns = allow_override if allow_override is not None else default_allow_patterns
        try:
            path = snapshot_download(
                repo_id=repo_id,
                revision=args.revision,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_override,
            )
        except Exception as exc:
            print(f"Failed to download {name} from {repo_id}: {exc}")
            print("If the repository is gated/private, run: huggingface-cli login")
            raise SystemExit(1) from exc

        candidates = checkpoint_candidates(Path(path), local_dir)
        manifest["teachers"][name] = {
            "source_repo": repo_id,
            "revision": args.revision or "main",
            "local_path": str(local_dir),
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_override,
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


def checkpoint_candidates(download_path: Path, local_dir: Path) -> list[str]:
    """List checkpoint candidates relative to the local teacher directory."""
    root = download_path.expanduser().resolve()
    base = local_dir.expanduser().resolve()
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
            continue
        resolved = path.resolve()
        try:
            candidates.append(str(resolved.relative_to(base)))
        except ValueError:
            candidates.append(str(resolved))
    return sorted(candidates)


def _split_patterns(value: str | None) -> list[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
