#!/usr/bin/env bash
# Orchestrator for Phase 15: Oracle-Guided Layer Dropping
set -e

# Setup environment for nohup compatibility
REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODELS=("mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B")
N_DROP=4
EVAL_SAMPLES=128

echo "================================================================"
echo "Phase 15: Oracle-Guided Layer Dropping (N=$N_DROP)"
echo "Starting at $(date)"
echo "================================================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing $MODEL"
    echo "========================================"
    
    # 1. Eval drop structural damage
    $VENV/bin/python -m code.phases.phase15_layer_drop.1_layer_drop_eval \
        --model "$MODEL" \
        --n_drop $N_DROP \
        --eval_samples $EVAL_SAMPLES
        
    # 2. Heal with Surgical LoRA (Boundary-Focused)
    $VENV/bin/python -m code.phases.phase15_layer_drop.2_heal_dropped_layers \
        --model "$MODEL" \
        --n_drop $N_DROP \
        --eval_samples $EVAL_SAMPLES \
        --train_samples 500 \
        --healing_mode boundary
done

echo ""
echo "================================================================"
echo "Phase 15 Complete at $(date)"
echo "================================================================"
