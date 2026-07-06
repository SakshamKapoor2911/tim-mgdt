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
STOP_AFTER_PHASE20_FAILURE="${STOP_AFTER_PHASE20_FAILURE:-1}"
MIN_PHASE20_STRICT_CONTINUE="${MIN_PHASE20_STRICT_CONTINUE:-0.30}"

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

has_bad_phase20_eval() {
    "$VENV/bin/python" - "$OUT_DIR" "$MIN_PHASE20_STRICT_CONTINUE" <<'PY'
import glob
import json
import os
import sys

out_dir, threshold_text = sys.argv[1], sys.argv[2]
threshold = float(threshold_text)
for path in sorted(glob.glob(os.path.join(out_dir, "gsm8k_eval_mask_guided_*.json"))):
    with open(path) as f:
        data = json.load(f)
    gsm8k = data.get("results", {}).get("gsm8k", {})
    strict = gsm8k.get("exact_match,strict-match")
    if strict is not None and strict < threshold:
        print(f"{os.path.basename(path)} strict={strict:.4f} < {threshold:.4f}")
        sys.exit(0)
sys.exit(1)
PY
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
    EVAL_PATTERN="$OUT_DIR/gsm8k_eval_mask_guided_${STRATEGY}_${TAG}*.json"

    echo
    if [[ "$FORCE_PHASE20" != "1" && "$STOP_AFTER_PHASE20_FAILURE" == "1" ]] && bad_eval="$(has_bad_phase20_eval)"; then
        echo "Stopping Phase 20 sweep before tag=$TAG; prior eval failed continuation gate: $bad_eval"
        break
    fi

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
    if [[ "$FORCE_PHASE20" != "1" ]] && compgen -G "$EVAL_PATTERN" >/dev/null; then
        echo "Skipping GSM8K eval for tag=$TAG; found existing eval matching $EVAL_PATTERN"
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
