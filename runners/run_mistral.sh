#!/usr/bin/env bash
# =============================================================================
# TIM-MGDT Phase 10 - Mistral-7B Specific Runner
# Pins to GPU 0, activates venv, and runs the evaluation with optimized settings
# =============================================================================

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====================================================================="
echo "🚀 Starting Mistral-7B Phase 10 Evaluation"
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

# 2. Pin to GPU 0
export CUDA_VISIBLE_DEVICES=0
echo "✓ Pinned execution to GPU 0 (CUDA_VISIBLE_DEVICES=0)"

# 3. Run evaluation
echo "⏳ Running Mistral-7B real evaluation (Large dataset, 16x batching)..."
python "$REPO_ROOT/tests/test_phase10_mistral7b_real.py"

echo "====================================================================="
echo "✓ Mistral-7B Evaluation Finished!"
echo "====================================================================="
