#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

MODEL="google/gemma-2-2b"
N_DROP=4

echo "========================================="
echo "Phase 15: Gemma-2-2B Baseline & Dropped Eval"
echo "========================================="
$VENV/bin/python -m code.phases.phase15_layer_drop.1_layer_drop_eval \
    --model "$MODEL" \
    --n_drop $N_DROP \
    --eval_samples 128
    
echo "Gemma evaluation complete."
