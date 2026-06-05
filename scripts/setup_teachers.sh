#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p external checkpoints/teachers/medical_sam3 checkpoints/teachers/cinema

if [ ! -d "external/Medical-SAM3/.git" ]; then
  git clone https://github.com/AIM-Research-Lab/Medical-SAM3 external/Medical-SAM3
else
  echo "external/Medical-SAM3 already exists; skipping clone."
fi

if [ ! -d "external/CineMA/.git" ]; then
  git clone https://github.com/mathpluscode/CineMA external/CineMA
else
  echo "external/CineMA already exists; skipping clone."
fi

echo "Teacher repositories are available under external/."
