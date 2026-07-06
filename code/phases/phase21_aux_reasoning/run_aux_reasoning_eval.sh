#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase21_aux_reasoning}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_SHORT="${MODEL##*/}"
MASK_TAG="${MASK_TAG:-s20}"
TASKS="${TASKS:-gsm8k_platinum_cot,bbh_cot_fewshot_multistep_arithmetic_two}"
LIMIT="${LIMIT:-100}"
EVAL_GPU="${EVAL_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
RUN_LABEL="${RUN_LABEL:-$(date -u +%Y-%m-%dT%H-%M-%S)}"
MODELS="${MODELS:-dense ${MASK_TAG}}"
FORCE="${FORCE:-0}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"
cd "$REPO"

model_args_for() {
    local tag="$1"
    if [[ "$tag" == "dense" ]]; then
        echo "pretrained=$MODEL"
        return
    fi

    local ckpt="$REPO/experiments/phase19/${MODEL_SHORT}_mask_pruned_${tag}"
    if [[ ! -d "$ckpt" ]]; then
        echo "Missing checkpoint: $ckpt" >&2
        exit 1
    fi
    echo "pretrained=$ckpt"
}

echo "=== Phase 21 auxiliary reasoning eval starting at $(date) ==="
echo "Model: $MODEL"
echo "Compared models: $MODELS"
echo "Tasks: $TASKS"
echo "Limit: $LIMIT"
echo "GPU: $EVAL_GPU"
echo "Batch size: $BATCH_SIZE"
echo "Run label: $RUN_LABEL"

for MODEL_TAG in $MODELS; do
    OUT_FILE="$OUT_DIR/eval_${MODEL_TAG}_${RUN_LABEL}.json"
    if [[ -f "$OUT_FILE" && "$FORCE" != "1" ]]; then
        echo "Skipping existing result: $OUT_FILE"
        continue
    fi

    echo
    echo "=== Evaluating $MODEL_TAG on $TASKS ==="
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV/bin/lm_eval" run --model hf \
        --model_args "$(model_args_for "$MODEL_TAG")" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --limit "$LIMIT" \
        --output_path "$OUT_FILE" \
        --log_samples
done

python scripts/summarize_phase21_aux.py \
    --exp-dir "$OUT_DIR" \
    --output "$OUT_DIR/summary_${RUN_LABEL}.json"

echo "=== Phase 21 auxiliary reasoning eval done at $(date) ==="
