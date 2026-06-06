#!/usr/bin/env python3
"""Prune unused CineMA checkpoint weights after an accidental full download."""

from __future__ import annotations

import argparse
from pathlib import Path


WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
DEFAULT_KEEP = {
    "finetuned/segmentation/acdc_sax/acdc_sax_0.safetensors",
    "finetuned/segmentation/acdc_sax/config.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune unused CineMA weights")
    parser.add_argument("--cinema_dir", default="checkpoints/teachers/cinema")
    parser.add_argument(
        "--keep",
        default=",".join(sorted(DEFAULT_KEEP)),
        help="Comma-separated relative paths to keep.",
    )
    parser.add_argument(
        "--keep_all_acdc_seeds",
        action="store_true",
        help="Keep acdc_sax_0/1/2.safetensors plus config.yaml.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cinema_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"CineMA directory not found: {root}")

    keep = {item.strip() for item in str(args.keep).split(",") if item.strip()}
    if args.keep_all_acdc_seeds:
        keep = {"finetuned/segmentation/acdc_sax/config.yaml"}
        keep.update(f"finetuned/segmentation/acdc_sax/acdc_sax_{seed}.safetensors" for seed in range(3))

    delete_files = []
    keep_files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in keep:
            keep_files.append(path)
            continue
        if path.suffix.lower() in WEIGHT_EXTENSIONS:
            delete_files.append(path)

    delete_bytes = sum(path.stat().st_size for path in delete_files)
    keep_bytes = sum(path.stat().st_size for path in keep_files if path.exists())
    print(f"CineMA dir: {root}")
    print("Keeping:")
    for item in sorted(keep):
        path = root / item
        status = "FOUND" if path.exists() else "MISSING"
        print(f"  [{status}] {item}")
    print(f"Kept selected files: {len(keep_files)} ({_fmt_bytes(keep_bytes)})")
    print(f"Weight files to delete: {len(delete_files)} ({_fmt_bytes(delete_bytes)})")
    for path in delete_files[:50]:
        print(f"  delete {path.relative_to(root).as_posix()}")
    if len(delete_files) > 50:
        print(f"  ... {len(delete_files) - 50} more")

    if not args.execute:
        print("Dry-run only. Add --execute to delete these weight files.")
        return

    for path in delete_files:
        path.unlink()
    remove_empty_dirs(root)
    print(f"Deleted {len(delete_files)} weight files and reclaimed {_fmt_bytes(delete_bytes)}.")


def remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


if __name__ == "__main__":
    main()
