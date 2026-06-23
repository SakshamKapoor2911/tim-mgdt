#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 16 Width Pruning Ablation starting at $(date) ===" 

for STRATEGY in ceil floor; do
    echo "Running width pruning with $STRATEGY strategy..."
    $VENV/bin/python -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "$STRATEGY" \
        --eval_samples 128
done

echo "=== Phase 16 Width Pruning Ablation done at $(date) ==="
