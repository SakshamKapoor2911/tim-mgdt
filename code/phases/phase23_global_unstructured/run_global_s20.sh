#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase23_global_unstructured}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_SHORT="${MODEL##*/}"
SPARSITY="${SPARSITY:-0.20}"
TAG="${TAG:-global_s20}"
C4_SAMPLES="${C4_SAMPLES:-64}"
PRUNE_GPU="${PRUNE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
RUN_LABEL="${RUN_LABEL:-full_$(date -u +%Y-%m-%dT%H-%M-%S)}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
FORCE="${FORCE:-0}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"
cd "$REPO"

MASK_META="$OUT_DIR/${MODEL_SHORT}_mask_pruning_${TAG}.json"
MASK_CKPT="$OUT_DIR/${MODEL_SHORT}_mask_pruned_${TAG}"
OUT_FILE="$OUT_DIR/gsm8k_eval_${TAG}_${RUN_LABEL}.json"

echo "=== Phase 23 global unstructured magnitude baseline starting at $(date) ==="
echo "Model: $MODEL"
echo "Sparsity: $SPARSITY"
echo "Tag: $TAG"
echo "C4 samples: $C4_SAMPLES"
echo "Prune GPU: $PRUNE_GPU"
echo "Eval GPU: $EVAL_GPU"
echo "Run label: $RUN_LABEL"

if [[ ! -f "$MASK_META" || ! -d "$MASK_CKPT" || "$FORCE" == "1" ]]; then
    echo
    echo "=== Building global unstructured magnitude checkpoint ==="
    CUDA_VISIBLE_DEVICES="$PRUNE_GPU" "$VENV/bin/python" -u -m code.phases.phase19.mask_pruning \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --mask_scope global \
        --eval_samples "$C4_SAMPLES" \
        --output_dir "$OUT_DIR" \
        --output_suffix "$TAG"
else
    echo "Reusing existing global checkpoint: $MASK_CKPT"
fi

LIMIT_ARGS=()
if [[ -n "$GSM8K_LIMIT" ]]; then
    LIMIT_ARGS=(--limit "$GSM8K_LIMIT")
fi

if [[ ! -f "$OUT_FILE" || "$FORCE" == "1" ]]; then
    echo
    echo "=== Evaluating global unstructured magnitude checkpoint on GSM8K ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
        --model_args "pretrained=$MASK_CKPT" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        "${LIMIT_ARGS[@]}" \
        --output_path "$OUT_FILE" \
        --log_samples
else
    echo "Skipping existing eval: $OUT_FILE"
fi

echo "=== Phase 23 global unstructured magnitude baseline done at $(date) ==="
