#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
WAIT_SESSIONS="${WAIT_SESSIONS:-phase19_s30_full phase19_s40_full phase19_adaptive_boundary}"
LOG_DIR="${LOG_DIR:-$REPO/experiments/phase20_mask_guided/run_logs}"

mkdir -p "$LOG_DIR"
cd "$REPO"

echo "=== Phase 20 queued runner starting at $(date) ==="
echo "Waiting for sessions: $WAIT_SESSIONS"

for SESSION in $WAIT_SESSIONS; do
    while tmux has-session -t "$SESSION" 2>/dev/null; do
        echo "Still waiting for tmux session $SESSION at $(date)"
        sleep 300
    done
done

echo "Phase 19 sessions finished at $(date)"
echo "=== Running Phase 20 mask-guided structured sweep ==="
bash code/phases/phase20_mask_guided/run_mask_guided_width_sweep.sh

echo "=== Phase 20 queued runner done at $(date) ==="
