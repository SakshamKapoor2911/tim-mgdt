#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_SHORT="${MODEL##*/}"
STRATEGY="${STRATEGY:-ceil}"
MASK_SPARSITY="${MASK_SPARSITY:-0.20}"
STRUCTURED_SPARSITIES="${STRUCTURED_SPARSITIES:-0.005 0.01 0.02 0.05}"
PRUNE_GPUS="${PRUNE_GPUS:-0,1}"
EVAL_GPU="${EVAL_GPU:-1}"
C4_SAMPLES="${C4_SAMPLES:-64}"
GSM8K_LIMIT="${GSM8K_LIMIT:-50}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase20_mask_guided}"
FORCE_PHASE20="${FORCE_PHASE20:-0}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"
cd "$REPO"

tag_from_sparsity() {
    local value="$1"
    local tag

    if [[ "$value" == 0.* ]]; then
        tag="${value#0.}"
        while [[ "${#tag}" -lt 2 ]]; do
            tag="${tag}0"
        done
        echo "s${tag}"
        return
    fi

    tag="${value//./p}"
    echo "s${tag}"
}

echo "=== Phase 20 mask-guided width sweep starting at $(date) ==="
echo "Model: $MODEL"
echo "Strategy: $STRATEGY"
echo "Scoring mask sparsity: $MASK_SPARSITY"
echo "Structured sparsities: $STRUCTURED_SPARSITIES"
echo "C4 samples: $C4_SAMPLES"
echo "GSM8K limit: $GSM8K_LIMIT"

for SPARSITY in $STRUCTURED_SPARSITIES; do
    TAG="$(tag_from_sparsity "$SPARSITY")"
    CKPT="$OUT_DIR/${MODEL_SHORT}_mask_guided_width_${STRATEGY}_${TAG}"
    RESULT_JSON="$OUT_DIR/${MODEL_SHORT}_mask_guided_width_${STRATEGY}_${TAG}.json"
    EVAL_JSON="$OUT_DIR/gsm8k_eval_mask_guided_${STRATEGY}_${TAG}.json"

    echo
    echo "=== Mask-guided structured prune sparsity=$SPARSITY tag=$TAG ==="
    if [[ "$FORCE_PHASE20" != "1" && -f "$RESULT_JSON" && -d "$CKPT" ]]; then
        echo "Skipping prune for tag=$TAG; found $RESULT_JSON and $CKPT"
    else
        CUDA_VISIBLE_DEVICES="$PRUNE_GPUS" "$VENV/bin/python" -u -m code.phases.phase20_mask_guided.mask_guided_width_pruning \
            --model "$MODEL" \
            --structured_sparsity "$SPARSITY" \
            --mask_sparsity "$MASK_SPARSITY" \
            --strategy "$STRATEGY" \
            --eval_samples "$C4_SAMPLES" \
            --output_dir "$OUT_DIR" \
            --output_suffix "$TAG"
    fi

    echo "=== GSM8K smoke eval tag=$TAG checkpoint=$CKPT ==="
    if [[ "$FORCE_PHASE20" != "1" && -f "$EVAL_JSON" ]]; then
        echo "Skipping GSM8K eval for tag=$TAG; found $EVAL_JSON"
    else
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
            --model_args "pretrained=$CKPT" \
            --tasks gsm8k \
            --num_fewshot 5 \
            --batch_size auto:4 \
            --limit "$GSM8K_LIMIT" \
            --output_path "$EVAL_JSON" \
            --log_samples
    fi
done

echo "=== Phase 20 mask-guided width sweep done at $(date) ==="
