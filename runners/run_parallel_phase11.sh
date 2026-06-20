#!/bin/bash
# =====================================================================
# Phase 11: Real Pruning Pipeline - PARALLEL Runner
# Expected runtime: ~30-60 minutes across TWO RTX 3090s
# =====================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "====================================================================="
echo "  Phase 11: PARALLEL Llama-3 & Mistral-7B Real Pruning Pipeline"
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

mkdir -p experiments/phase11

echo "[START] Launching Llama-3-8B on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m code.phases.phase11.run_full_pipeline \
    --model llama-3-8b \
    --samples 128 \
    --ppl-threshold 5.0 \
    --output-dir experiments/phase11 > experiments/phase11/llama_run.log 2>&1 &
PID_LLAMA=$!
echo "  -> Llama PID: $PID_LLAMA"

echo "[START] Launching Mistral-7B on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -m code.phases.phase11.run_full_pipeline \
    --model mistral-7b \
    --samples 128 \
    --ppl-threshold 5.0 \
    --output-dir experiments/phase11 > experiments/phase11/mistral_run.log 2>&1 &
PID_MISTRAL=$!
echo "  -> Mistral PID: $PID_MISTRAL"

echo ""
echo "[INFO] Both processes are now running in parallel in the background."
echo "[INFO] Logs are being written to:"
echo "       - experiments/phase11/llama_run.log"
echo "       - experiments/phase11/mistral_run.log"
echo ""
echo "[WAIT] Waiting for both Phase 11 pipelines to complete (this will take 30-60 mins)..."

# Tail logs in the background so the user sees some output if they run it interactively
tail -f experiments/phase11/llama_run.log experiments/phase11/mistral_run.log &
TAIL_PID=$!

# Wait for both main processes to finish
# Note: we temporarily disable 'set -e' so 'wait' returning non-zero doesn't kill the script immediately
set +e
wait $PID_LLAMA
LLAMA_STATUS=$?

wait $PID_MISTRAL
MISTRAL_STATUS=$?
set -e

# Stop tailing
kill $TAIL_PID 2>/dev/null || true

echo ""
echo "====================================================================="
if [ $LLAMA_STATUS -eq 0 ] && [ $MISTRAL_STATUS -eq 0 ]; then
    echo "[DONE] Both models completed successfully! Results in experiments/phase11/"
else
    echo "[ERROR] One or both models failed!"
    echo "  Llama exit code: $LLAMA_STATUS"
    echo "  Mistral exit code: $MISTRAL_STATUS"
fi
echo "====================================================================="
