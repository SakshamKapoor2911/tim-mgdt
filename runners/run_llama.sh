#!/usr/bin/env bash
# =============================================================================
# TIM-MGDT Phase 10 - Llama-3-8B Specific Runner
# Pins to GPU 1, activates venv, and runs the evaluation with optimized settings
# =============================================================================

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====================================================================="
echo "🚀 Starting Llama-3-8B Phase 10 Evaluation"
echo "====================================================================="

# 1. Activate virtual environment
if [ -d "$REPO_ROOT/venv" ]; then
    echo "✓ Activating virtual environment: $REPO_ROOT/venv"
    source "$REPO_ROOT/venv/bin/activate"
elif [ -d "$REPO_ROOT/.venv" ]; then
    echo "✓ Activating virtual environment: $REPO_ROOT/.venv"
    source "$REPO_ROOT/.venv/bin/activate"
else
    echo "⚠ Virtual environment (venv or .venv) not found. Attempting run with system python."
fi

# 2. Pin to GPU 1
export CUDA_VISIBLE_DEVICES=1
echo "✓ Pinned execution to GPU 1 (CUDA_VISIBLE_DEVICES=1)"

# 3. Run evaluation
echo "⏳ Running Llama-3-8B real evaluation (Large dataset, 16x batching)..."
python "$REPO_ROOT/tests/test_phase10_llama3_8b_real.py"

echo "====================================================================="
echo "✓ Llama-3-8B Evaluation Finished!"
echo "====================================================================="
