#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
WAIT_SESSIONS="${WAIT_SESSIONS:-phase19_s30_full phase19_s40_full phase19_adaptive_boundary phase20_mask_guided_queue}"

cd "$REPO"

echo "=== Experiment publish queue starting at $(date) ==="
echo "Waiting for sessions: $WAIT_SESSIONS"

for SESSION in $WAIT_SESSIONS; do
    while tmux has-session -t "$SESSION" 2>/dev/null; do
        echo "Still waiting for tmux session $SESSION at $(date)"
        sleep 300
    done
done

echo "All waited sessions finished at $(date)"
bash scripts/publish_experiment_updates.sh
echo "=== Experiment publish queue done at $(date) ==="
