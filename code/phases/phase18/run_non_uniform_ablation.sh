#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 18 Non-Uniform Pruning starting at $(date) ===" 

# 1. Prune the model using dynamic sensitivity-guided allocation
echo "Running sensitivity-guided non-uniform width pruning (Global Sparsity: 20%, Strategy: ceil)..."
CUDA_VISIBLE_DEVICES=1 $VENV/bin/python -u -m code.phases.phase18.non_uniform_pruning \
    --model "$MODEL" \
    --global_sparsity 0.20 \
    --strategy "ceil"

# 2. Evaluate the pruned model on GSM8K
echo "Running GSM8K evaluation on non-uniformly pruned checkpoint (Limit 50)..."
CUDA_VISIBLE_DEVICES=1 $VENV/bin/lm_eval --model hf \
    --model_args pretrained=$REPO/experiments/phase18/Mistral-7B-v0.1_non_uniform_pruned_ceil \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --limit 50 \
    --output_path $REPO/experiments/phase18/gsm8k_eval_non_uniform_ceil.json \
    --log_samples

echo "=== Phase 18 Non-Uniform Pruning done at $(date) ==="
