#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
STRATEGY="${STRATEGY:-ceil}"
PRUNE_GPUS="${PRUNE_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"
C4_SAMPLES="${C4_SAMPLES:-64}"
GSM8K_LIMIT="${GSM8K_LIMIT:-50}"
OUT_DIR="$REPO/experiments/phase18b"
MODEL_SHORT="${MODEL##*/}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"

echo "=== Phase 18B structural width sweep starting at $(date) ==="
echo "Model: $MODEL"
echo "Strategy: $STRATEGY"
echo "C4 samples: $C4_SAMPLES"
echo "GSM8K limit: $GSM8K_LIMIT"

for SPARSITY in 0.05 0.10 0.15 0.20; do
    case "$SPARSITY" in
        0.05) TAG="s05" ;;
        0.10) TAG="s10" ;;
        0.15) TAG="s15" ;;
        0.20) TAG="s20" ;;
        *) echo "Unsupported sparsity tag: $SPARSITY" >&2; exit 1 ;;
    esac

    echo
    echo "=== Structural prune sparsity=$SPARSITY tag=$TAG ==="
    CUDA_VISIBLE_DEVICES="$PRUNE_GPUS" "$VENV/bin/python" -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --strategy "$STRATEGY" \
        --eval_samples "$C4_SAMPLES" \
        --output_dir "$OUT_DIR" \
        --output_suffix "$TAG"

    CKPT="$OUT_DIR/${MODEL_SHORT}_pruned_${STRATEGY}_${TAG}"
    echo "=== GSM8K smoke eval tag=$TAG checkpoint=$CKPT ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" --model hf \
        --model_args "pretrained=$CKPT" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size auto:4 \
        --limit "$GSM8K_LIMIT" \
        --output_path "$OUT_DIR/gsm8k_eval_${STRATEGY}_${TAG}.json" \
        --log_samples
done

echo "=== Phase 18B structural width sweep done at $(date) ==="
