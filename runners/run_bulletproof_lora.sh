#!/usr/bin/env bash
set -e

REPO=/home/skapoor/tim-mgdt
VENV=$REPO/venv

export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODELS=("mistralai/Mistral-7B-v0.1" "meta-llama/Meta-Llama-3-8B")

echo "=== bulletproof_lora starting at $(date) ===" 

for MODEL in "${MODELS[@]}"; do
    echo "Running bulletproof LoRA eval for $MODEL"
    $VENV/bin/python -u -m code.phases.phase14_final.5_run_bulletproof_lora \
        --model "$MODEL" \
        --eval_samples 256 \
        --train_samples 1000 \
        --rank 16
done

echo "=== bulletproof_lora done at $(date) ==="
