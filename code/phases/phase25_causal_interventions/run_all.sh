#!/usr/bin/env bash
#
# Master execution script for Phase 25: Causal Interventions
# Sequence: P1 (Procrustes + Rotation) -> P2 (Negative Controls) -> P3 (Metrics) -> P4 (Location Sweep)
#
# Usage:
#   tmux new-session -s phase25
#   ./code/phases/phase25_causal_interventions/run_all.sh 2>&1 | tee experiments/phase25_causal_interventions/run_all.log
#
# Each stage writes a checkpoint file on completion and skips if already done.
# Kill-safe: re-run to continue from last completed stage.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/phase25_causal_interventions}"
VENV="${VENV:-$REPO/venv}"
NUM_EVAL="${NUM_EVAL:-100}"
NUM_CALIB="${NUM_CALIB:-50}"
GPU="${GPU:-0}"

mkdir -p "$OUTPUT_DIR"

echo "=== Phase 25: Causal Interventions ==="
echo "  REPO:          $REPO"
echo "  MODEL:         $MODEL"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "  NUM_EVAL:      $NUM_EVAL"
echo "  NUM_CALIB:     $NUM_CALIB"
echo "  GPU:           $GPU"
echo "  Started at:    $(date)"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Stage 1: P1 — Procrustes Realignment + Artificial Rotation Sweep
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_1_complete.json" ]; then
    echo "[SKIP] P1 already complete."
else
    echo "[RUN]  P1: Procrustes Realignment + Artificial Rotation"
    echo "       Est. GPU time: ~6h (Procrustes 45min + 5-angle sweep ~5h)"
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p1_procrustes_rotation.log"
    echo "[DONE] P1 complete at $(date)"
fi

# ---------------------------------------------------------------------------
# Stage 2: P2 — Negative Controls
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_2_complete.json" ]; then
    echo "[SKIP] P2 already complete."
else
    echo "[RUN]  P2: Negative Controls (Random Rotation, Wrong-Layer, Wrong-PC)"
    echo "       Est. GPU time: ~3h (3 controls × 1h each)"
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.2_run_negative_controls \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p2_negative_controls.log"
    echo "[DONE] P2 complete at $(date)"
fi

# ---------------------------------------------------------------------------
# Stage 3: P3 — Competing Metrics Matrix (no GPU eval needed after collect)
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_3_complete.json" ]; then
    echo "[SKIP] P3 already complete."
else
    echo "[RUN]  P3: Fixed-Depth Competing Metrics Matrix"
    echo "       Est. GPU time: ~1h (metric computation, no GSM8K eval)"
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.3_run_competing_metrics \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p3_competing_metrics.log"
    touch "$OUTPUT_DIR/_checkpoint_3_complete.json"
    echo "[DONE] P3 complete at $(date)"
fi

# ---------------------------------------------------------------------------
# Stage 4: P4 — Multi-Location & Seed Robustness Sweep
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_4_complete.json" ]; then
    echo "[SKIP] P4 already complete."
else
    echo "[RUN]  P4: Multi-Location & Seed Sweep (Early/Mid/Late × 3 seeds)"
    echo "       Est. GPU time: ~9h (3 locations × 3 seeds × 1h each)"
    "$VENV/bin/python" -m code.phases.phase25_causal_interventions.4_run_multi_location_sweep \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_seeds 3 \
        --seed_base 0 2>&1 | tee "$OUTPUT_DIR/p4_location_sweep.log"
    echo "[DONE] P4 complete at $(date)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 25 Complete ==="
echo "  Finished at: $(date)"
echo "  Results in:  $OUTPUT_DIR"
echo ""
echo "Result files:"
ls -lh "$OUTPUT_DIR"/exp_*.json 2>/dev/null || echo "  (no result files found)"
