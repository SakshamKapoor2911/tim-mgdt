#!/usr/bin/env bash
# =====================================================================
# Master Execution Script for NeurReps 2026 Archival Submission
#
# Workflows:
#   1. Backup existing N=100 results to *_n100.json
#   2. Multi-GPU FIM Pre-Pruning (CUDA_VISIBLE_DEVICES=0,1) for Ceil and Floor strategies
#   3. Dual-GPU Split Execution:
#      - GPU 0: Full Phase 25 (1A+1B + P4 Location Sweeps) at N=500 + 4-way McNemar
#      - GPU 1: lm_eval GSM8K 8-shot (N=512 limit) for Baseline, Ceil, Floor
#   4. Selective Post-Execution Copy to *_n500.json
# =====================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/experiments/phase25_causal_interventions}"
PHASE16_DIR="${PHASE16_DIR:-$REPO/experiments/phase16}"
VENV="${VENV:-$REPO/venv}"
NUM_EVAL="${NUM_EVAL:-500}"
NUM_CALIB="${NUM_CALIB:-50}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUTPUT_DIR" "$PHASE16_DIR" "$REPO/logs"

log_gpu() {
    local label="${1:-}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
            | while IFS=, read -r idx used total; do
                echo "  [GPU${idx}${label:+, }${label}] ${used}, ${total}"
            done
    fi
}

echo "====================================================================="
echo "  NeurReps 2026 Archival Master Pipeline Launch"
echo "  Model:       $MODEL"
echo "  Num Eval:    $NUM_EVAL"
echo "  Output Dir:  $OUTPUT_DIR"
echo "  Started at:  $(date)"
echo "====================================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Backup existing N=100 JSON result files
# ---------------------------------------------------------------------------
echo "[STEP 1] Backing up existing N=100 result JSON files..."
for f in exp_1a_procrustes exp_1b_sufficiency exp_1b_rotation_sweep exp_4_location_sweep exp_mcnemar; do
    if [ -f "$OUTPUT_DIR/${f}.json" ]; then
        cp -v "$OUTPUT_DIR/${f}.json" "$OUTPUT_DIR/${f}_n100.json"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Multi-GPU FIM Pre-Pruning (Sequential on CUDA_VISIBLE_DEVICES=0,1)
# ---------------------------------------------------------------------------
echo ""
echo "[STEP 2] Running Multi-GPU FIM Width Pruning (Ceil + Floor)..."
log_gpu "before pruning"

if [ -d "$PHASE16_DIR/Mistral-7B-v0.1_pruned_ceil" ]; then
    echo "[SKIP] Ceil pruned model already exists at $PHASE16_DIR/Mistral-7B-v0.1_pruned_ceil"
else
    echo "[PRUNE] Running Ceil strategy width pruning on CUDA_VISIBLE_DEVICES=0,1..."
    CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "ceil" \
        --eval_samples 128 \
        --output_dir "$PHASE16_DIR" 2>&1 | tee "$REPO/logs/width_pruning_ceil_$(date +%Y%m%d).log"
fi

if [ -d "$PHASE16_DIR/Mistral-7B-v0.1_pruned_floor" ]; then
    echo "[SKIP] Floor pruned model already exists at $PHASE16_DIR/Mistral-7B-v0.1_pruned_floor"
else
    echo "[PRUNE] Running Floor strategy width pruning on CUDA_VISIBLE_DEVICES=0,1..."
    CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" -u -m code.phases.phase16.width_pruning \
        --model "$MODEL" \
        --sparsity 0.20 \
        --strategy "floor" \
        --eval_samples 128 \
        --output_dir "$PHASE16_DIR" 2>&1 | tee "$REPO/logs/width_pruning_floor_$(date +%Y%m%d).log"
fi

log_gpu "after pruning"

# ---------------------------------------------------------------------------
# Step 3: Launch Dual-GPU Split Execution (GPU 0: Phase 25, GPU 1: lm_eval)
# ---------------------------------------------------------------------------
echo ""
echo "[STEP 3] Launching Dual-GPU Split Execution..."

# --- GPU 0 Session (Phase 25 Full 1A+1B + P4 Location Sweeps at N=500) ---
echo "[GPU 0] Launching Phase 25 (Full 1A+1B + P4 Location Sweeps)..."
(
    export CUDA_VISIBLE_DEVICES=0
    
    # Run Full 1A + 1B at N=500 to evaluate all 4 conditions on identical 500 prompts
    echo "[GPU 0] Running 1_run_procrustes_and_rotation.py (1A+1B) at N=$NUM_EVAL..."
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$REPO/logs/p1_full_n500.log"
        
    # Run P4 Multi-Location Sweep on GPU 0
    echo "[GPU 0] Running P4 Location Sweep at N=$NUM_EVAL..."
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.4_run_multi_location_sweep \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_seeds 3 --seed_base 0 \
        --locations early middle late 2>&1 | tee "$REPO/logs/p4_location_sweep_n500.log"

    touch "$OUTPUT_DIR/_checkpoint_phase25_n500_complete.json"
    echo "[GPU 0] Phase 25 N=500 complete!"
) &
PID_GPU0=$!

# --- GPU 1 Session (lm_eval GSM8K 8-shot for Baseline, Ceil, Floor) ---
echo "[GPU 1] Launching lm_eval GSM8K 8-shot (Limit=512)..."
(
    export CUDA_VISIBLE_DEVICES=1
    
    echo "[GPU 1] lm_eval Baseline model..."
    "$VENV/bin/lm_eval" --model hf \
        --model_args pretrained="$MODEL" \
        --tasks gsm8k \
        --num_fewshot 8 \
        --limit 512 \
        --batch_size auto:4 \
        --output_path "$PHASE16_DIR/gsm8k_eval_baseline_n512.json" 2>&1 | tee "$REPO/logs/gsm8k_baseline_n512.log"
        
    if [ -d "$PHASE16_DIR/Mistral-7B-v0.1_pruned_ceil" ]; then
        echo "[GPU 1] lm_eval Ceil pruned model..."
        "$VENV/bin/lm_eval" --model hf \
            --model_args pretrained="$PHASE16_DIR/Mistral-7B-v0.1_pruned_ceil" \
            --tasks gsm8k \
            --num_fewshot 8 \
            --limit 512 \
            --batch_size auto:4 \
            --output_path "$PHASE16_DIR/gsm8k_eval_ceil_n512.json" 2>&1 | tee "$REPO/logs/gsm8k_ceil_n512.log"
    fi

    if [ -d "$PHASE16_DIR/Mistral-7B-v0.1_pruned_floor" ]; then
        echo "[GPU 1] lm_eval Floor pruned model..."
        "$VENV/bin/lm_eval" --model hf \
            --model_args pretrained="$PHASE16_DIR/Mistral-7B-v0.1_pruned_floor" \
            --tasks gsm8k \
            --num_fewshot 8 \
            --limit 512 \
            --batch_size auto:4 \
            --output_path "$PHASE16_DIR/gsm8k_eval_floor_n512.json" 2>&1 | tee "$REPO/logs/gsm8k_floor_n512.log"
    fi

    touch "$PHASE16_DIR/_checkpoint_eval_n512_complete.json"
    echo "[GPU 1] lm_eval N=512 complete!"
) &
PID_GPU1=$!

wait $PID_GPU0
wait $PID_GPU1

# ---------------------------------------------------------------------------
# Step 4: Selective copy of regenerated N=500 outputs to *_n500.json
# ---------------------------------------------------------------------------
echo ""
echo "[STEP 4] Copying regenerated N=500 outputs to explicit *_n500.json files..."
for f in exp_1a_procrustes exp_1b_sufficiency exp_1b_rotation_sweep exp_4_location_sweep exp_mcnemar; do
    if [ -f "$OUTPUT_DIR/${f}.json" ]; then
        cp -v "$OUTPUT_DIR/${f}.json" "$OUTPUT_DIR/${f}_n500.json"
    fi
done

echo "====================================================================="
echo "  NeurReps 2026 Master Pipeline Successfully Completed!"
echo "  Finished at: $(date)"
echo "====================================================================="
