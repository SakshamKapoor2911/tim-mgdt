#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

MODEL="mistralai/Mistral-7B-v0.1"
N_DROP=4
THICKNESSES=(1 2 3)

echo "Starting Phase 16C (Boundary Thickness Ablation) for $MODEL (drop $N_DROP)..."
for THICKNESS in "${THICKNESSES[@]}"; do
    echo "Healing with boundary thickness = $THICKNESS"
    $VENV/bin/python -m code.phases.phase15_layer_drop.2_heal_dropped_layers \
        --model "$MODEL" \
        --n_drop $N_DROP \
        --eval_samples 128 \
        --train_samples 1000 \
        --rank 16 \
        --healing_mode boundary \
        --boundary_thickness $THICKNESS
done
echo "Phase 16C complete."
