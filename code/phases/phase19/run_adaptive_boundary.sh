#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
WAIT_SESSIONS="${WAIT_SESSIONS:-phase19_s30_full phase19_s40_full}"
S30_STRONG_MIN="${S30_STRONG_MIN:-0.35}"
LOWER_SPARSITY="${LOWER_SPARSITY:-0.25}"
UPPER_SPARSITY="${UPPER_SPARSITY:-0.35}"
RUN_LABEL="${RUN_LABEL:-full_$(date -u +%Y-%m-%dT%H-%M-%S)}"
LOG_DIR="${LOG_DIR:-$REPO/experiments/phase19/run_logs}"

mkdir -p "$LOG_DIR"
cd "$REPO"

echo "=== Phase 19 adaptive boundary runner starting at $(date) ==="
echo "Waiting for sessions: $WAIT_SESSIONS"

for SESSION in $WAIT_SESSIONS; do
    while tmux has-session -t "$SESSION" 2>/dev/null; do
        echo "Still waiting for tmux session $SESSION at $(date)"
        sleep 300
    done
done

echo "Prior full-eval sessions finished at $(date)"

read -r NEXT_SPARSITY S30_STRICT < <(
    python - <<'PY'
import glob
import json
import os
import sys


def metric(path):
    with open(path) as f:
        data = json.load(f)
    gsm8k = data.get("results", {}).get("gsm8k", {})
    samples = data.get("n-samples", {}).get("gsm8k", {})
    return {
        "path": path,
        "n": samples.get("effective", gsm8k.get("sample_len")) or 0,
        "strict": gsm8k.get("exact_match,strict-match"),
    }


paths = glob.glob("experiments/phase19/gsm8k_eval_mask_s30_full*.json")
if not paths:
    print("missing_s30_result 0")
    sys.exit(0)

best = sorted((metric(p) for p in paths), key=lambda m: (m["n"], os.path.getmtime(m["path"])))[-1]
strict = best.get("strict")
if strict is None:
    print("missing_s30_metric 0")
    sys.exit(0)

threshold = float(os.environ.get("S30_STRONG_MIN", "0.35"))
lower = os.environ.get("LOWER_SPARSITY", "0.25")
upper = os.environ.get("UPPER_SPARSITY", "0.35")
print((upper if strict >= threshold else lower), strict)
PY
)

if [[ "$NEXT_SPARSITY" == missing_* ]]; then
    echo "Could not choose next sparsity: $NEXT_SPARSITY"
    exit 1
fi

echo "s30 strict EM: $S30_STRICT"
echo "Next boundary sparsity: $NEXT_SPARSITY"

export REPO
export SPARSITIES="$NEXT_SPARSITY"
export GSM8K_LIMIT="${GSM8K_LIMIT:-50}"
export C4_SAMPLES="${C4_SAMPLES:-64}"
export PRUNE_GPUS="${PRUNE_GPUS:-0,1}"
export EVAL_GPU="${EVAL_GPU:-1}"

echo "=== Building mask checkpoint and smoke eval for sparsity=$NEXT_SPARSITY ==="
bash code/phases/phase19/run_mask_sweep.sh

echo "=== Running full GSM8K eval for sparsity=$NEXT_SPARSITY ==="
RUN_LABEL="$RUN_LABEL" EVAL_GPU="${FULL_EVAL_GPU:-0}" SPARSITIES="$NEXT_SPARSITY" \
    bash code/phases/phase19/run_full_gsm8k_eval.sh

echo "=== Phase 19 adaptive boundary runner done at $(date) ==="
python scripts/summarize_phase19.py
