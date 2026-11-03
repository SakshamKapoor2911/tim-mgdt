#!/usr/bin/env bash
set -e

REPO="/home/skapoor/tim-mgdt"
VENV="$REPO/venv"
export PYTHONPATH="$REPO:$PYTHONPATH"

MODEL="mistralai/Mistral-7B-v0.1"

echo "=== Smart Parallel Ablation (HF) ==="

# 1. Prune models using BOTH GPUs (prevents OOM during FIM gradient calculation)
echo "Running width pruning with ceil strategy (Multi-GPU)..."
CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/python -u -m code.phases.phase16.width_pruning \
    --model "$MODEL" \
    --sparsity 0.20 \
    --strategy "ceil" \
    --eval_samples 128

echo "Running width pruning with floor strategy (Multi-GPU)..."
CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/python -u -m code.phases.phase16.width_pruning \
    --model "$MODEL" \
    --sparsity 0.20 \
    --strategy "floor" \
    --eval_samples 128

# 2. Evaluate Baseline, Ceil, and Floor optimally
# GPU 0: Baseline (90 mins), then Ceil (90 mins) = 180 mins
# GPU 1: Floor (90 mins)

echo "[GPU 0] Starting Baseline evaluation..."
(
    export CUDA_VISIBLE_DEVICES=0
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$MODEL \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_baseline.json \
        --log_samples
        
    echo "[GPU 0] Starting Ceil evaluation..."
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$REPO/experiments/phase16/Mistral-7B-v0.1_pruned_ceil \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_ceil.json \
        --log_samples
) &
PID1=$!

echo "[GPU 1] Starting Floor evaluation..."
(
    export CUDA_VISIBLE_DEVICES=1
    $VENV/bin/lm_eval --model hf \
        --model_args pretrained=$REPO/experiments/phase16/Mistral-7B-v0.1_pruned_floor \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --output_path $REPO/experiments/phase16/gsm8k_eval_floor.json \
        --log_samples
) &
PID2=$!

wait $PID1
wait $PID2

echo "=== Smart Parallel Ablation Done ==="
