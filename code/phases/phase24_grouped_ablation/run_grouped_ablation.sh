#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
VENV="${VENV:-$REPO/venv}"
OUT_DIR="${OUT_DIR:-$REPO/experiments/phase24_grouped_ablation}"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_SHORT="${MODEL##*/}"
SPARSITY="${SPARSITY:-0.25}"
C4_SAMPLES="${C4_SAMPLES:-32}"
GSM8K_LIMIT="${GSM8K_LIMIT:-300}"
BUILD_GPU="${BUILD_GPU:-0}"
EVAL_GPUS="${EVAL_GPUS:-0 1}"
BATCH_SIZE="${BATCH_SIZE:-auto:4}"
RUN_LABEL="${RUN_LABEL:-limit${GSM8K_LIMIT}_$(date -u +%Y-%m-%dT%H-%M-%S)}"
FORCE="${FORCE:-0}"
RUN_MODULE_CONTINGENCY="${RUN_MODULE_CONTINGENCY:-auto}"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$OUT_DIR"
cd "$REPO"

read -r -a GPU_LIST <<< "$EVAL_GPUS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "EVAL_GPUS must contain at least one GPU id" >&2
    exit 2
fi

echo "=== Phase 24 grouped ablation starting at $(date) ==="
echo "Model: $MODEL"
echo "Base mask: layerwise magnitude at sparsity=$SPARSITY"
echo "C4 samples: $C4_SAMPLES"
echo "GSM8K limit: $GSM8K_LIMIT"
echo "Build GPU: $BUILD_GPU"
echo "Eval GPUs: $EVAL_GPUS"
echo "Run label: $RUN_LABEL"

build_one() {
    local tag="$1"
    local meta="$OUT_DIR/${MODEL_SHORT}_grouped_mask_${tag}.json"
    local ckpt="$OUT_DIR/${MODEL_SHORT}_mask_pruned_${tag}"
    if [[ "$FORCE" != "1" && -f "$meta" && -d "$ckpt" ]]; then
        echo "Reusing checkpoint for $tag: $ckpt"
        return
    fi

    echo
    echo "=== Building grouped checkpoint: $tag ==="
    CUDA_VISIBLE_DEVICES="$BUILD_GPU" "$VENV/bin/python" -u \
        -m code.phases.phase24_grouped_ablation.build_grouped_mask \
        --model "$MODEL" \
        --sparsity "$SPARSITY" \
        --ablation "$tag" \
        --eval_samples "$C4_SAMPLES" \
        --output_dir "$OUT_DIR"
}

eval_one() {
    local tag="$1"
    local gpu="$2"
    local ckpt="$OUT_DIR/${MODEL_SHORT}_mask_pruned_${tag}"
    local eval_dir="$OUT_DIR/$tag"
    local out_file="$eval_dir/gsm8k_eval_${tag}_${RUN_LABEL}.json"

    mkdir -p "$eval_dir"
    if [[ "$FORCE" != "1" ]] && compgen -G "$eval_dir/gsm8k_eval_${tag}_${RUN_LABEL}*.json" >/dev/null; then
        echo "Skipping existing eval for $tag: $(compgen -G "$eval_dir/gsm8k_eval_${tag}_${RUN_LABEL}*.json" | sort | tail -1)"
        return
    fi
    if [[ ! -d "$ckpt" ]]; then
        echo "Missing checkpoint for $tag: $ckpt" >&2
        exit 1
    fi

    echo
    echo "=== Evaluating $tag on GPU $gpu ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$VENV/bin/lm_eval" run --model hf \
        --model_args "pretrained=$ckpt" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size "$BATCH_SIZE" \
        --limit "$GSM8K_LIMIT" \
        --output_path "$out_file" \
        --log_samples
}

eval_batch() {
    local tags=("$@")
    local pids=()
    local fail=0
    local gpu_idx=0

    for tag in "${tags[@]}"; do
        local gpu="${GPU_LIST[$gpu_idx]}"
        eval_one "$tag" "$gpu" &
        pids+=("$!")
        gpu_idx=$((gpu_idx + 1))

        if [[ "$gpu_idx" -ge "${#GPU_LIST[@]}" ]]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || fail=1
            done
            pids=()
            gpu_idx=0
        fi
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    if [[ "$fail" != "0" ]]; then
        echo "At least one eval failed" >&2
        exit 1
    fi
}

summarize() {
    local summary_json="$OUT_DIR/phase24_summary_${RUN_LABEL}.json"
    local summary_md="$OUT_DIR/phase24_summary_${RUN_LABEL}.md"
    "$VENV/bin/python" -u -m code.phases.phase24_grouped_ablation.bootstrap_gsm8k_ci \
        --result_glob "$OUT_DIR/*/gsm8k_eval_*_${RUN_LABEL}*.json" \
        --output_dir "$OUT_DIR" \
        --output "$summary_json" \
        --markdown "$summary_md"
}

decision_from_summary() {
    local summary_json="$OUT_DIR/phase24_summary_${RUN_LABEL}.json"
    "$VENV/bin/python" - <<PY
import json
with open("$summary_json") as f:
    print(json.load(f)["decision"]["recommendation"])
PY
}

LAYER_TAGS=(revert_early revert_mid revert_late)
MODULE_TAGS=(revert_mlp revert_attention)

for tag in "${LAYER_TAGS[@]}"; do
    build_one "$tag"
done

eval_batch "${LAYER_TAGS[@]}"
summarize

DECISION="$(decision_from_summary)"
echo "Layer-block decision: $DECISION"

if [[ "$RUN_MODULE_CONTINGENCY" == "1" || ( "$RUN_MODULE_CONTINGENCY" == "auto" && "$DECISION" == "run_module_contingency" ) ]]; then
    echo
    echo "=== Running MLP-vs-attention contingency ==="
    for tag in "${MODULE_TAGS[@]}"; do
        build_one "$tag"
    done
    eval_batch "${MODULE_TAGS[@]}"
    summarize
else
    echo "Skipping module contingency."
fi

echo "=== Phase 24 grouped ablation done at $(date) ==="
