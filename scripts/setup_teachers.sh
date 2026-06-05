#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TEACHER="${1:-both}"
case "$TEACHER" in
  medical_sam3|cinema|both) ;;
  *)
    echo "Usage: bash scripts/setup_teachers.sh [medical_sam3|cinema|both]"
    exit 2
    ;;
esac

mkdir -p external checkpoints/teachers/medical_sam3 checkpoints/teachers/cinema

clone_teacher_repo() {
  local url="$1"
  local dest="$2"

  if [ -d "$dest/.git" ]; then
    echo "$dest already exists as a git repo; skipping clone."
    return
  fi

  if [ -d "$dest" ]; then
    local non_placeholder_count
    non_placeholder_count="$(find "$dest" -mindepth 1 ! -name ".gitkeep" | wc -l | tr -d ' ')"
    if [ "$non_placeholder_count" = "0" ]; then
      rm -rf "$dest"
    else
      echo "ERROR: $dest exists but is not a git repo and is not just a .gitkeep placeholder."
      echo "Move it away or remove it, then rerun:"
      echo "  mv $dest ${dest}.backup"
      echo "  bash scripts/setup_teachers.sh"
      exit 1
    fi
  fi

  git clone "$url" "$dest"
}

if [ "$TEACHER" = "medical_sam3" ] || [ "$TEACHER" = "both" ]; then
  clone_teacher_repo https://github.com/AIM-Research-Lab/Medical-SAM3 external/Medical-SAM3
fi

if [ "$TEACHER" = "cinema" ] || [ "$TEACHER" = "both" ]; then
  clone_teacher_repo https://github.com/mathpluscode/CineMA external/CineMA
fi

echo "Teacher repositories are available under external/."
