#!/usr/bin/env bash
# =============================================================================
# TIM-MGDT Phase 10 - Parallel Dual-GPU Runner
# Launches both Mistral-7B (GPU 0) and Llama-3-8B (GPU 1) in parallel with venv
# =============================================================================

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "====================================================================="
echo "🚀 Starting Parallel Dual-GPU Phase 10 Evaluations"
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

# 2. Execute parallel python script
echo "⚡ Launching both models simultaneously in parallel subprocesses..."
python "$REPO_ROOT/tests/run_parallel_gpus.py"

echo "====================================================================="
echo "✓ Parallel GPU Run Complete!"
echo "====================================================================="
