#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 16 Width Pruning Ablation starting at $(date) ===" 

echo "Running GSM8K evaluation on Baseline $MODEL..."
$VENV/bin/lm_eval --model hf \
    --model_args pretrained=$MODEL \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --output_path $REPO/experiments/phase16/gsm8k_eval_baseline.json \
    --log_samples

for STRATEGY in ceil floor; do
    echo "Running width pruning with $STRATEGY strategy..."
    $VENV/bin/python -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "$STRATEGY" \
        --eval_samples 128
        
    echo "Running GSM8K evaluation on $STRATEGY checkpoint..."
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$REPO/experiments/phase16/Mistral-7B-v0.1_pruned_$STRATEGY \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_${STRATEGY}.json \
        --log_samples
done

echo "=== Phase 16 Width Pruning Ablation done at $(date) ==="
