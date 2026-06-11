#!/usr/bin/env bash
# One-shot environment setup for the MolmoMotion-1M data-generation pipeline.
# Run from the repo root inside a fresh Python 3.12 environment (conda or venv).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

echo "[2/4] Installing SAM 3 (third_party/sam3)..."
pip install -e third_party/sam3

echo "[3/4] Installing ViPE (third_party/vipe; compiles a CUDA extension)..."
# Requires nvcc on PATH and a matching CUDA toolkit (12.x). See third_party/vipe/README.md.
pip install -e third_party/vipe

echo "[4/4] Downloading the AllTracker checkpoint..."
bash scripts/download_models.sh

echo
echo "Done. Verify with:  python scripts/check_env.py"
