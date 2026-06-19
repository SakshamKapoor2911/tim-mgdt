#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

MODELS=("mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B")

echo "Starting Phase 16A..."
for MODEL in "${MODELS[@]}"; do
    echo "Running 16A for $MODEL"
    $VENV/bin/python -m code.phases.phase15_layer_drop.2_heal_dropped_layers \
        --model "$MODEL" \
        --n_drop 4 \
        --eval_samples 128 \
        --train_samples 2000 \
        --rank 32 \
        --eval_interval 250 \
        --eval_interval_samples 16 \
        --healing_mode boundary
done
echo "Phase 16A complete."
