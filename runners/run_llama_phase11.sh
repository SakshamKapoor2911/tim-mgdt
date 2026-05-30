#!/bin/bash
# =====================================================================
# Phase 11: Real Pruning Pipeline - Llama-3-8B Runner
# Expected runtime: ~30-60 minutes on RTX 3090
# =====================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "====================================================================="
echo "  Phase 11: Llama-3-8B Real Pruning Pipeline"
echo "====================================================================="

# Detect virtual environment
if [ -d "$REPO_ROOT/venv" ]; then
    echo "[OK] Virtual environment found: venv"
    source "$REPO_ROOT/venv/bin/activate"
elif [ -d "$REPO_ROOT/.venv" ]; then
    echo "[OK] Virtual environment found: .venv"
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "[WARN] No virtual environment found. Using system python."
fi

# Ensure datasets is installed
pip install -q datasets 2>/dev/null || echo "[WARN] Could not install datasets. Make sure it's available."

# Pin to GPU 0
export CUDA_VISIBLE_DEVICES=0

echo "[START] Running Llama-3-8B real pruning pipeline..."
echo "[NOTE] This will take 30-60 minutes on a single RTX 3090."
python -m code.phases.phase11.run_full_pipeline \
    --model llama-3-8b \
    --samples 128 \
    --ppl-threshold 5.0 \
    --output-dir experiments/phase11

echo "====================================================================="
echo "[DONE] Llama-3-8B Phase 11 complete! Results in experiments/phase11/"
echo "====================================================================="
