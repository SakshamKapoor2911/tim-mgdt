#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

MODEL="mistralai/Mistral-7B-v0.1"
N_DROPS=(2 6 8)

echo "Starting Phase 16B (Pareto Sweep) for $MODEL..."
for N_DROP in "${N_DROPS[@]}"; do
    echo "Evaluating drop $N_DROP"
    $VENV/bin/python -m code.phases.phase15_layer_drop.1_layer_drop_eval \
        --model "$MODEL" \
        --n_drop $N_DROP \
        --eval_samples 128
        
    echo "Healing drop $N_DROP"
    $VENV/bin/python -m code.phases.phase15_layer_drop.2_heal_dropped_layers \
        --model "$MODEL" \
        --n_drop $N_DROP \
        --eval_samples 128 \
        --train_samples 1000 \
        --rank 16 \
        --healing_mode boundary
done
echo "Phase 16B complete."
