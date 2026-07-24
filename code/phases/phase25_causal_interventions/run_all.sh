#!/usr/bin/env bash
#
# Master execution script for Phase 25: Causal Interventions
# DUAL-GPU version: runs P2+P3 in parallel, P4 split across GPUs.
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUTPUT_DIR"

log_gpu() {
    local label="${1:-}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null \
            | while IFS=, read -r idx used total; do
                echo "  [GPU${idx}${label:+, }${label}] ${used}, ${total}"
            done
    fi
}

echo "=== Phase 25: Causal Interventions (DUAL GPU) ==="
echo "  MODEL:         $MODEL"
echo "  NUM_EVAL:      $NUM_EVAL"
echo "  Started at:    $(date)"
echo ""
log_gpu "start"

# ---------------------------------------------------------------------------
# Stage 1: P1 — Procrustes Realignment + Sufficiency (GPU 0 only, sequential)
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_1_complete.json" ]; then
    echo "[SKIP] P1 already complete."
else
    echo "[RUN]  P1: Procrustes (R*) + Sufficiency ((R*)^T) on GPU 0"
    CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.1_run_procrustes_and_rotation \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p1_procrustes_rotation.log"
    echo "[DONE] P1 complete at $(date)"
    log_gpu "after P1"
fi

# ---------------------------------------------------------------------------
# Stage 2+3: P2 (GPU 0) + P3 (GPU 1) in PARALLEL
# ---------------------------------------------------------------------------
p2_pid=""
p3_pid=""
p2_failed=""
p3_failed=""

if [ -f "$OUTPUT_DIR/_checkpoint_2_complete.json" ]; then
    echo "[SKIP] P2 already complete."
else
    echo "[RUN]  P2: Negative Controls on GPU 0 (est. ~3h)"
    sleep 3
    log_gpu "before P2"
    (CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.2_run_negative_controls \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p2_negative_controls.log") &
    p2_pid=$!
fi

if [ -f "$OUTPUT_DIR/_checkpoint_3_complete.json" ]; then
    echo "[SKIP] P3 already complete."
else
    echo "[RUN]  P3: Competing Metrics on GPU 1 (est. ~1h)"
    sleep 3
    log_gpu "before P3"
    (CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.3_run_competing_metrics \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_calibration "$NUM_CALIB" 2>&1 | tee "$OUTPUT_DIR/p3_competing_metrics.log") &
    p3_pid=$!
fi

# Wait for both P2 and P3 to finish
if [ -n "$p2_pid" ] || [ -n "$p3_pid" ]; then
    echo "[WAIT] Waiting for P2 (GPU 0) and P3 (GPU 1)..."
    if [ -n "$p2_pid" ]; then wait "$p2_pid" && { touch "$OUTPUT_DIR/_checkpoint_2_complete.json"; echo "[DONE] P2 complete at $(date)"; } || p2_failed="P2"; fi
    if [ -n "$p3_pid" ]; then wait "$p3_pid" && { touch "$OUTPUT_DIR/_checkpoint_3_complete.json"; echo "[DONE] P3 complete at $(date)"; } || p3_failed="P3"; fi
    if [ -n "$p2_failed$p3_failed" ]; then
        echo "[FAIL] Failed stages: ${p2_failed} ${p3_failed}"
        exit 1
    fi
    log_gpu "after P2+P3"
fi

# ---------------------------------------------------------------------------
# Stage 4: P4 — Multi-Location Sweep split across BOTH GPUs
# ---------------------------------------------------------------------------
if [ -f "$OUTPUT_DIR/_checkpoint_4_complete.json" ]; then
    echo "[SKIP] P4 already complete."
else
    echo "[RUN]  P4: Location Sweep split across GPUs"
    echo "       GPU 0: early+late, GPU 1: middle (est. ~4.5h total)"
    sleep 3
    log_gpu "before P4"

    p4_gpu0_pid=""
    p4_gpu1_pid=""
    p4_failed=""

    # GPU 0: early + late (6 evals)
    (CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.4_run_multi_location_sweep \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_seeds 3 --seed_base 0 \
        --locations early late \
        --output_suffix "_gpu0" 2>&1 | tee "$OUTPUT_DIR/p4_location_sweep_gpu0.log") &
    p4_gpu0_pid=$!

    # GPU 1: middle (3 evals)
    (CUDA_VISIBLE_DEVICES=1 "$VENV/bin/python" -m code.phases.phase25_causal_interventions.4_run_multi_location_sweep \
        --model_name "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_eval_samples "$NUM_EVAL" \
        --num_seeds 3 --seed_base 0 \
        --locations middle \
        --output_suffix "_gpu1" 2>&1 | tee "$OUTPUT_DIR/p4_location_sweep_gpu1.log") &
    p4_gpu1_pid=$!

    # Wait for both
    if [ -n "$p4_gpu0_pid" ]; then wait "$p4_gpu0_pid" || p4_failed="GPU0"; fi
    if [ -n "$p4_gpu1_pid" ]; then wait "$p4_gpu1_pid" || p4_failed="GPU1"; fi

    if [ -n "$p4_failed" ]; then
        echo "[FAIL] P4 on $p4_failed failed."
        exit 1
    fi

    # Merge GPU 0 + GPU 1 results
    echo "[MERGE] Merging P4 results from both GPUs..."
    "$VENV/bin/python" -c "
import json, os
d = os.path.abspath('$OUTPUT_DIR')
merged = {'experiment': 'P4_multi_location_seed_sweep_merged', 'model': '$MODEL',
          'num_seeds': $NUM_EVAL, 'locations': {}}
for suffix in ['_gpu0', '_gpu1']:
    fp = os.path.join(d, f'exp_4_location_sweep{suffix}.json')
    if os.path.exists(fp):
        with open(fp) as f:
            data = json.load(f)
            merged['locations'].update(data.get('locations', {}))
if merged['locations']:
    with open(os.path.join(d, 'exp_4_location_sweep.json'), 'w') as f:
        json.dump(merged, f, indent=2)
    print('[MERGE] Merged into exp_4_location_sweep.json')
    for loc, res in merged['locations'].items():
        print(f'  {res[\"location\"]}: {res[\"mean\"]:.2f}% [95% CI: {res[\"clopper_pearson_ci_95\"][0]:.2f}, {res[\"clopper_pearson_ci_95\"][1]:.2f}]')
else:
    print('[MERGE] No result files found')
" 2>&1 | tee -a "$OUTPUT_DIR/p4_location_sweep.log"

    touch "$OUTPUT_DIR/_checkpoint_4_complete.json"
    echo "[DONE] P4 complete at $(date)"
    log_gpu "after P4"
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
