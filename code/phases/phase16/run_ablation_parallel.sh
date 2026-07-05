#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Phase 16 Width Pruning Ablation (Parallel) starting at $(date) ==="

# Task 1: Baseline -> Floor on GPU 0
(
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU 0] Running GSM8K evaluation on Baseline $MODEL..."
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$MODEL \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_baseline.json \
        --log_samples
    
    echo "[GPU 0] Running width pruning with floor strategy..."
    $VENV/bin/python -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "floor" \
        --eval_samples 128
        
    echo "[GPU 0] Running GSM8K evaluation on floor checkpoint..."
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$REPO/experiments/phase16/Mistral-7B-v0.1_pruned_floor \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_floor.json \
        --log_samples
) &
PID1=$!

# Task 2: Ceil on GPU 1
(
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU 1] Running width pruning with ceil strategy..."
    $VENV/bin/python -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "ceil" \
        --eval_samples 128
        
    echo "[GPU 1] Running GSM8K evaluation on ceil checkpoint..."
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$REPO/experiments/phase16/Mistral-7B-v0.1_pruned_ceil \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_ceil.json \
        --log_samples
) &
PID2=$!

wait $PID1
wait $PID2

echo "=== Phase 16 Width Pruning Ablation (Parallel) done at $(date) ==="
