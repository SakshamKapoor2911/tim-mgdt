#!/usr/bin/env bash
#
# Sequence runner for experiments to run after Phase 25 main pipeline.
# Waits for _checkpoint_2_complete.json and _checkpoint_4_complete.json,
# then runs P5 (PC₁-specific un-rotation).
#
# Usage:
#   ./code/phases/phase25_causal_interventions/run_next.sh 2>&1 | tee experiments/phase25_causal_interventions/run_next.log
#
# Kill-safe: skips completed stages via checkpoint markers.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/phase25_causal_interventions}"
VENV="${VENV:-$REPO/venv}"
NUM_EVAL="${NUM_EVAL:-100}"
NUM_CALIB="${NUM_CALIB:-50}"
GPU="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== Phase 25 Post-Pipeline: PC₁ Un-Rotation ==="
echo "  Started at: $(date)"
echo "  Waiting for main pipeline (P2+P4) to complete..."

# Wait for main pipeline checkpoints
while [ ! -f "$OUTPUT_DIR/_checkpoint_2_complete.json" ] || \
      [ ! -f "$OUTPUT_DIR/_checkpoint_4_complete.json" ]; do
    sleep 60
    echo "  [$(date)] Waiting for checkpoints... (P2: $([ -f "$OUTPUT_DIR/_checkpoint_2_complete.json" ] && echo done || echo wait), P4: $([ -f "$OUTPUT_DIR/_checkpoint_4_complete.json" ] && echo done || echo wait))"
done

echo "  Main pipeline complete at $(date)"

# Run P5: PC₁-specific un-rotation
if [ -f "$OUTPUT_DIR/_checkpoint_5_complete.json" ]; then
    echo "[SKIP] P5 already complete."
else
    echo "[RUN]  P5: PC₁-Specific Subspace Un-Rotation"
    echo "       Est. GPU time: ~1h (collect + SVD + 3 evals)"
    sleep 3
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1
    fi

    CUDA_VISIBLE_DEVICES="$GPU" "$VENV/bin/python" -m code.phases.phase25_causal_interventions.5_pc1_unrotation \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p5_pc1_unrotation.log"

    touch "$OUTPUT_DIR/_checkpoint_5_complete.json"
    echo "[DONE] P5 complete at $(date)"
fi

echo ""
echo "=== Phase 25 All Experiments Complete ==="
echo "  Finished at: $(date)"
echo "  Results in:  $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/exp_*.json 2>/dev/null || echo "  (no result files)"
