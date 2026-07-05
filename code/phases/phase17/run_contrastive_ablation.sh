#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 17 Contrastive Pruning starting at $(date) ===" 

# 1. Prune the model using contrastive activation tracing
echo "Running contrastive circuit sparing (Sparsity: 20%, Strategy: ceil)..."
CUDA_VISIBLE_DEVICES=0 $VENV/bin/python -u -m code.phases.phase17.contrastive_pruning \
    --model "$MODEL" \
    --sparsity 0.20 \
    --strategy "ceil"

# 2. Evaluate the pruned model on GSM8K
echo "Running GSM8K evaluation on contrastively pruned checkpoint..."
CUDA_VISIBLE_DEVICES=0 $VENV/bin/lm_eval --model hf \
    --model_args pretrained=$REPO/experiments/phase17/Mistral-7B-v0.1_contrastive_pruned_ceil \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size auto:4 \
    --limit 50 \
    --output_path $REPO/experiments/phase17/gsm8k_eval_contrastive_ceil.json \
    --log_samples

echo "=== Phase 17 Contrastive Pruning done at $(date) ==="
