#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

clone_teacher_repo https://github.com/AIM-Research-Lab/Medical-SAM3 external/Medical-SAM3
clone_teacher_repo https://github.com/mathpluscode/CineMA external/CineMA

echo "Teacher repositories are available under external/."
