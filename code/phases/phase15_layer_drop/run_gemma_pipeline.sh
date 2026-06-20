#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1

MODEL="google/gemma-2-2b"
N_DROP=4

echo "========================================="
echo "Step 1: Phase 14 - Causal Sensitivity Oracle for $MODEL"
echo "========================================="
$VENV/bin/python -m code.phases.phase14_final.1_measure_sensitivity \
    --model "$MODEL" \
    --samples 128

echo "========================================="
echo "Step 2: Phase 15 - Baseline vs Dropped Eval"
echo "========================================="
# The eval_dropped_model script is not strictly necessary if we just run the healed script,
# but we need the baseline/dropped ppls. Let's run a quick eval if possible.
# For simplicity, 2_heal_dropped_layers will evaluate the healed model.
# Let's run the phase16c sweep directly, which also does healing.
# Note: 2_heal_dropped_layers reads baseline/dropped from a file. If it's missing, it just leaves them null.
# So we can just go straight to healing.

echo "========================================="
echo "Step 3: Phase 16C - Boundary Thickness Ablation"
echo "========================================="
THICKNESSES=(1 2 3)
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

echo "Gemma-2-2B complete."
