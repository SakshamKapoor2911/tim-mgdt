#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase19}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_SHORT="${MODEL##*/}"
SPARSITIES="${SPARSITIES:-0.20}"
EVAL_GPU="${EVAL_GPU:-0}"
RUN_LABEL="${RUN_LABEL:-full_$(date -u +%Y-%m-%dT%H-%M-%S)}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

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

mkdir -p "$OUT_DIR"

echo "=== Phase 19 full GSM8K eval starting at $(date) ==="
echo "Model: $MODEL"
echo "Sparsities: $SPARSITIES"
echo "GPU: $EVAL_GPU"
echo "Batch size: $BATCH_SIZE"
echo "Run label: $RUN_LABEL"

for SPARSITY in $SPARSITIES; do
    TAG="$(tag_from_sparsity "$SPARSITY")"
    CKPT="$OUT_DIR/${MODEL_SHORT}_mask_pruned_${TAG}"
    OUT_FILE="$OUT_DIR/gsm8k_eval_mask_${TAG}_${RUN_LABEL}.json"

    if [[ ! -d "$CKPT" ]]; then
        echo "Missing checkpoint: $CKPT" >&2
        exit 1
    fi

    LIMIT_ARGS=()
    if [[ -n "$GSM8K_LIMIT" ]]; then
        LIMIT_ARGS=(--limit "$GSM8K_LIMIT")
    fi

    echo
    echo "=== GSM8K eval sparsity=$SPARSITY tag=$TAG checkpoint=$CKPT ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
        --model_args "pretrained=$CKPT" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        "${LIMIT_ARGS[@]}" \
        --output_path "$OUT_FILE" \
        --log_samples
done

echo "=== Phase 19 full GSM8K eval done at $(date) ==="
