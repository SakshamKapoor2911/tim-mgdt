#!/usr/bin/env bash
# =====================================================================
# NeurReps 2026 Full N=500 Pipeline (8-shot, unified protocol)
#
# GPU 0: P1 (sufficiency + random SO(d) control) -> P4 (location) -> P5 (PC1)
# GPU 1: P6 (causal PC ablation) -> P7 (skip-depth rotation curve)
# Then : energy collapse measurement (3 models, per-passage CI) on GPU 0
# =====================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/experiments/phase25_causal_interventions}"
VENV="${VENV:-$REPO/venv}"
NUM_EVAL="${NUM_EVAL:-500}"
NUM_CALIB="${NUM_CALIB:-50}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUTPUT_DIR" "$REPO/logs"

echo "====================================================================="
echo "  NeurReps 2026 Full N=500 Pipeline (8-shot unified protocol)"
echo "  Model: $MODEL | Num Eval: $NUM_EVAL | Calib: $NUM_CALIB"
echo "  Started at: $(date)"
echo "====================================================================="

log_gpu() {
    local label="${1:-}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
        | while IFS=, read -r idx used util; do
            echo "  [GPU${idx}${label:+, }${label}] ${used}, ${util}"
        done
}

# ---------------------------------------------------------------------------
# GPU 0 Session
# ---------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=0

    echo "[GPU 0] P1: Sufficiency + Random SO(d) control (N=$NUM_EVAL)..."
    "$VENV/bin/python" -u -m code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation \
        --model_name "$MODEL" --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" --num_calibration "$NUM_CALIB" \
        2>&1 | tee "$REPO/logs/p1_sufficiency_n500.log"
    log_gpu "after P1"

    echo "[GPU 0] P4: Multi-location sweep (N=$NUM_EVAL, 3 seeds)..."
    "$VENV/bin/python" -u -m code.phases.phase25_causal_interventions.4_run_multi_location_sweep \
        --model_name "$MODEL" --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" --num_seeds 3 --seed_base 0 \
        --locations early middle late \
        2>&1 | tee "$REPO/logs/p4_location_n500.log"
    log_gpu "after P4"

    echo "[GPU 0] P5: PC1 un-rotation + Procrustes + random controls (N=$NUM_EVAL)..."
    "$VENV/bin/python" -u -m code.phases.phase25_causal_interventions.5_pc1_unrotation \
        --model_name "$MODEL" --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" --num_calibration "$NUM_CALIB" \
        2>&1 | tee "$REPO/logs/p5_pc1_unrotation_n500.log"
    log_gpu "after P5"
) &
PID_GPU0=$!

# ---------------------------------------------------------------------------
# GPU 1 Session
# ---------------------------------------------------------------------------
(
    export CUDA_VISIBLE_DEVICES=1

    echo "[GPU 1] P6: Causal PC ablation at Layer 10 (N=$NUM_EVAL)..."
    "$VENV/bin/python" -u -m code.phases.phase25_causal_interventions.6_pc_ablation \
        --model_name "$MODEL" --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" --num_calibration "$NUM_CALIB" \
        2>&1 | tee "$REPO/logs/p6_pc_ablation_n500.log"
    log_gpu "after P6"

    echo "[GPU 1] P7: Skip-depth rotation curve k=0..4 (N=$NUM_EVAL)..."
    "$VENV/bin/python" -u -m code.phases.phase25_causal_interventions.7_rotation_curve \
        --model_name "$MODEL" --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" --num_calibration "$NUM_CALIB" \
        2>&1 | tee "$REPO/logs/p7_rotation_curve_n500.log"
    log_gpu "after P7"
) &
PID_GPU1=$!

wait $PID_GPU0
wait $PID_GPU1

# ---------------------------------------------------------------------------
# Energy collapse (GPU 0 now free; CPU-heavy SVD)
# ---------------------------------------------------------------------------
echo "Energy collapse measurement (3 models, per-passage CI)..."
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u "$REPO/scratch/compute_true_energy_collapse.py" \
    --num_prompts "$NUM_CALIB" --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$REPO/logs/energy_collapse.log"

echo "====================================================================="
echo "  NeurReps 2026 Full N=500 Pipeline COMPLETE at $(date)"
echo "  Outputs in: $OUTPUT_DIR"
echo "====================================================================="
