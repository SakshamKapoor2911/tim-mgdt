#!/bin/bash
# =====================================================================
# Phase 11: Real Pruning Pipeline - GPT-2 Runner (Fast Test)
# Expected runtime: ~5 minutes on GPU, ~15 minutes on CPU
# =====================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "====================================================================="
echo "  Phase 11: GPT-2 Real Pruning Pipeline"
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

# Use GPU 0 if available
export CUDA_VISIBLE_DEVICES=0

echo "[START] Running GPT-2 real pruning pipeline..."
python -m code.phases.phase11.run_full_pipeline \
    --model gpt2 \
    --samples 64 \
    --ppl-threshold 5.0 \
    --output-dir experiments/phase11

echo "====================================================================="
echo "[DONE] GPT-2 Phase 11 complete! Results in experiments/phase11/"
echo "====================================================================="
