#!/usr/bin/env bash
# =====================================================================
# Isolated Pre-Flight Smoke Test Suite for NeurReps 2026 Pipeline
# =====================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL="mistralai/Mistral-7B-v0.1"
SMOKE_DIR="$REPO/experiments/phase25_causal_interventions/_smoke"
SMOKE_P16_DIR="$REPO/experiments/phase16/_smoke"
SMOKE_EVAL="$REPO/experiments/phase16/_smoke_eval.json"
VENV="$REPO/venv"

mkdir -p "$SMOKE_DIR" "$SMOKE_P16_DIR"

echo "====================================================================="
echo "  NeurReps 2026 Isolated Pre-Flight Smoke Test"
echo "  Smoke Dir: $SMOKE_DIR"
echo "====================================================================="

# 1. Smoke test width pruning on dual GPU (isolated output dir)
echo "[1/3] Smoke testing width pruning (CUDA_VISIBLE_DEVICES=0,1)..."
CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" -u -m code.phases.phase16.width_pruning \
    --model "$MODEL" \
    --sparsity 0.20 \
    --strategy "ceil" \
    --eval_samples 10 \
    --output_dir "$SMOKE_P16_DIR"

# 2. Smoke test Phase 25 interventions (full 1A+1B in isolated _smoke dir)
echo "[2/3] Smoke testing Phase 25 full 1A+1B in isolated _smoke dir..."
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation \
    --model_name "$MODEL" \
    --output_dir "$SMOKE_DIR" \
    --num_eval_samples 10 \
    --num_calibration 10

# 3. Smoke test lm_eval harness flags
echo "[3/3] Smoke testing lm_eval harness..."
"$VENV/bin/lm_eval" --model hf \
    --model_args pretrained="$MODEL" \
    --tasks gsm8k \
    --num_fewshot 8 \
    --limit 10 \
    --batch_size auto:4 \
    --output_path "$SMOKE_EVAL"

echo "====================================================================="
echo "  Pre-Flight Smoke Test Cleanly Completed! No production files touched."
echo "====================================================================="
