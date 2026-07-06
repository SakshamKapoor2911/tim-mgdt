#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase22_cross_model}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_SHORT="${MODEL##*/}"
MODEL_TAG="${MODEL_TAG:-$(MODEL_SHORT="$MODEL_SHORT" python - <<'PY'
import os
import re

tag = re.sub(r"[^a-z0-9]+", "-", os.environ["MODEL_SHORT"].lower()).strip("-")
print(tag)
PY
)}"
SPARSITY="${SPARSITY:-0.20}"
MASK_TAG="${MASK_TAG:-s20}"
OUTPUT_SUFFIX="${MODEL_TAG}_${MASK_TAG}"
TASKS="${TASKS:-gsm8k}"
LIMIT="${LIMIT:-100}"
C4_SAMPLES="${C4_SAMPLES:-32}"
PRUNE_GPU="${PRUNE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
RUN_LABEL="${RUN_LABEL:-$(date -u +%Y-%m-%dT%H-%M-%S)}"
FORCE="${FORCE:-0}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"
cd "$REPO"

MASK_META="$OUT_DIR/${MODEL_SHORT}_mask_pruning_${OUTPUT_SUFFIX}.json"
MASK_CKPT="$OUT_DIR/${MODEL_SHORT}_mask_pruned_${OUTPUT_SUFFIX}"
DENSE_OUT="$OUT_DIR/eval_${MODEL_TAG}_dense_${RUN_LABEL}.json"
MASK_OUT="$OUT_DIR/eval_${MODEL_TAG}_${MASK_TAG}_${RUN_LABEL}.json"

echo "=== Phase 22 cross-model mask smoke starting at $(date) ==="
echo "Model: $MODEL"
echo "Model tag: $MODEL_TAG"
echo "Sparsity: $SPARSITY"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "Prune GPU: $PRUNE_GPU"
echo "Eval GPU: $EVAL_GPU"
echo "Run label: $RUN_LABEL"

if [[ ! -f "$MASK_META" || ! -d "$MASK_CKPT" || "$FORCE" == "1" ]]; then
    echo
    echo "=== Building 20% dense-shaped mask checkpoint ==="
    CUDA_VISIBLE_DEVICES="$PRUNE_GPU" "$VENV/bin/python" -u -m code.phases.phase19.mask_pruning \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --eval_samples "$C4_SAMPLES" \
        --output_dir "$OUT_DIR" \
        --output_suffix "$OUTPUT_SUFFIX"
else
    echo "Reusing existing mask checkpoint: $MASK_CKPT"
fi

if [[ ! -f "$DENSE_OUT" || "$FORCE" == "1" ]]; then
    echo
    echo "=== Evaluating dense model ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
        --model_args "pretrained=$MODEL" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --output_path "$DENSE_OUT" \
        --log_samples
else
    echo "Skipping existing dense eval: $DENSE_OUT"
fi

if [[ ! -f "$MASK_OUT" || "$FORCE" == "1" ]]; then
    echo
    echo "=== Evaluating masked model ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
        --model_args "pretrained=$MASK_CKPT" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --output_path "$MASK_OUT" \
        --log_samples
else
    echo "Skipping existing mask eval: $MASK_OUT"
fi

python scripts/summarize_phase22_cross_model.py \
    --exp-dir "$OUT_DIR" \
    --model-tag "$MODEL_TAG" \
    --output "$OUT_DIR/summary_${MODEL_TAG}_${RUN_LABEL}.json"

echo "=== Phase 22 cross-model mask smoke done at $(date) ==="
