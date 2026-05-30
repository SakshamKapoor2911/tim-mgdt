#!/bin/bash
# =====================================================================
# Phase 11: Advanced LLM Subnetwork Optimization - Llama-3-8B Runner
# =====================================================================

set -e

# Resolve repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "====================================================================="
echo "🚀 Starting Llama-3-8B Phase 11 Evaluation"
echo "====================================================================="

# Detect virtual environment
if [ -d "$REPO_ROOT/venv" ]; then
    echo "✓ Virtual environment found: venv"
    source "$REPO_ROOT/venv/bin/activate"
elif [ -d "$REPO_ROOT/.venv" ]; then
    echo "✓ Virtual environment found: .venv"
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "⚠ Virtual environment (venv or .venv) not found. Attempting run with system python."
fi

# Pin execution to GPU 1 (Second RTX 3090)
export CUDA_VISIBLE_DEVICES=1
echo "✓ Pinned execution to GPU 1 (CUDA_VISIBLE_DEVICES=1)"

echo "⏳ Running Llama-3-8B Phase 11 advanced pipeline..."
python -m code.phases.phase11.run_full_pipeline --model llama-3-8b --output-dir experiments/phase11

echo "====================================================================="
echo "✓ Llama-3-8B Phase 11 Finished!"
echo "====================================================================="
