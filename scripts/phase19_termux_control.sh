#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
LOG_DIR="${LOG_DIR:-$REPO/experiments/phase19/run_logs}"
STATUS_INTERVAL="${STATUS_INTERVAL:-60}"

mkdir -p "$LOG_DIR"
cd "$REPO"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/phase19_termux_control.sh status
  bash scripts/phase19_termux_control.sh ensure
  bash scripts/phase19_termux_control.sh queue-phase20
  bash scripts/phase19_termux_control.sh publish
  bash scripts/phase19_termux_control.sh queue-publish
  bash scripts/phase19_termux_control.sh monitor [seconds]
  bash scripts/phase19_termux_control.sh logs [s30|s40|adaptive|phase20|phase20-gpu1|phase21|publish]
  bash scripts/phase19_termux_control.sh attach [tmux-session-name]

Run this on the GPU host after SSHing in from Termux. Termux is the control
terminal; the CUDA jobs run on the remote GPU machine in tmux.
EOF
}

timestamp() {
    date -u +%Y-%m-%dT%H-%M-%S
}

session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

has_full_result() {
    local tag="$1"
    compgen -G "$REPO/experiments/phase19/gsm8k_eval_mask_${tag}_full*.json" >/dev/null
}

has_boundary_result() {
    compgen -G "$REPO/experiments/phase19/gsm8k_eval_mask_s25_full*.json" >/dev/null ||
        compgen -G "$REPO/experiments/phase19/gsm8k_eval_mask_s35_full*.json" >/dev/null
}

has_phase20_result() {
    compgen -G "$REPO/experiments/phase20_mask_guided/gsm8k_eval_mask_guided_*.json" >/dev/null
}

start_full_eval() {
    local session="$1"
    local tag="$2"
    local sparsity="$3"
    local gpu="$4"
    local stamp
    local log

    if session_exists "$session"; then
        echo "$session already running"
        return
    fi
    if has_full_result "$tag"; then
        echo "$tag full result already exists"
        return
    fi

    stamp="$(timestamp)"
    log="$LOG_DIR/gsm8k_${tag}_full_${stamp}.log"
    echo "starting $session on GPU $gpu; log: $log"
    tmux new-session -d -s "$session" \
        "cd '$REPO' && RUN_LABEL='full_${stamp}' EVAL_GPU='$gpu' SPARSITIES='$sparsity' GSM8K_LIMIT= bash code/phases/phase19/run_full_gsm8k_eval.sh > '$log' 2>&1"
}

start_adaptive() {
    local session="phase19_adaptive_boundary"
    local stamp
    local log

    if session_exists "$session"; then
        echo "$session already running"
        return
    fi
    if has_boundary_result; then
        echo "adaptive boundary full result already exists"
        return
    fi

    stamp="$(timestamp)"
    log="$LOG_DIR/adaptive_boundary_${stamp}.log"
    echo "starting $session; log: $log"
    tmux new-session -d -s "$session" \
        "cd '$REPO' && bash code/phases/phase19/run_adaptive_boundary.sh > '$log' 2>&1"
}

start_phase20_queue() {
    local session="phase20_mask_guided_queue"
    local log_dir="$REPO/experiments/phase20_mask_guided/run_logs"
    local stamp
    local log

    mkdir -p "$log_dir"
    if session_exists "$session"; then
        echo "$session already running"
        return
    fi
    if has_phase20_result && [[ "${FORCE_PHASE20:-0}" != "1" ]]; then
        echo "phase20 smoke results already exist; set FORCE_PHASE20=1 to queue again"
        return
    fi

    stamp="$(timestamp)"
    log="$log_dir/phase20_queue_${stamp}.log"
    echo "starting $session; log: $log"
    tmux new-session -d -s "$session" \
        "cd '$REPO' && bash code/phases/phase20_mask_guided/run_after_phase19.sh > '$log' 2>&1"
}

start_publish_queue() {
    local session="experiment_publish_queue"
    local log_dir="$REPO/experiments/publish_logs"
    local stamp
    local log

    mkdir -p "$log_dir"
    if session_exists "$session"; then
        echo "$session already running"
        return
    fi

    stamp="$(timestamp)"
    log="$log_dir/publish_queue_${stamp}.log"
    echo "starting $session; log: $log"
    tmux new-session -d -s "$session" \
        "cd '$REPO' && bash scripts/publish_after_sessions.sh > '$log' 2>&1"
}

status() {
    python scripts/phase19_status.py
}

ensure() {
    command -v tmux >/dev/null
    command -v python >/dev/null

    start_full_eval phase19_s30_full s30 0.30 0
    start_full_eval phase19_s40_full s40 0.40 1
    start_adaptive
    status
}

monitor() {
    local interval="${1:-$STATUS_INTERVAL}"
    while true; do
        printf '\n=== %s ===\n' "$(date)"
        status
        sleep "$interval"
    done
}

latest_log_for() {
    local target="$1"
    local pattern
    case "$target" in
        s30) pattern="$LOG_DIR/gsm8k_s30_full_*.log" ;;
        s40) pattern="$LOG_DIR/gsm8k_s40_full_*.log" ;;
        adaptive) pattern="$LOG_DIR/adaptive_boundary_*.log" ;;
        phase20) pattern="$REPO/experiments/phase20_mask_guided/run_logs/phase20_queue_*.log" ;;
        phase20-gpu1) pattern="$REPO/experiments/phase20_mask_guided/run_logs/phase20_gpu1_*.log" ;;
        phase21) pattern="$REPO/experiments/phase21_aux_reasoning/run_logs/aux_gpu1_*.log" ;;
        publish) pattern="$REPO/experiments/publish_logs/publish_queue_*.log" ;;
        *) return 1 ;;
    esac
    compgen -G "$pattern" | sort | tail -1
}

logs() {
    local target="${1:-adaptive}"
    local log
    log="$(latest_log_for "$target")"
    if [[ -z "$log" ]]; then
        echo "no log found for $target" >&2
        exit 1
    fi
    tail -n "${LINES:-80}" -f "$log"
}

attach() {
    local session="${1:-phase19_adaptive_boundary}"
    tmux attach -t "$session"
}

cmd="${1:-status}"
shift || true

case "$cmd" in
    status) status "$@" ;;
    ensure) ensure "$@" ;;
    queue-phase20) start_phase20_queue "$@" ;;
    publish) bash scripts/publish_experiment_updates.sh "$@" ;;
    queue-publish) start_publish_queue "$@" ;;
    monitor) monitor "$@" ;;
    logs) logs "$@" ;;
    attach) attach "$@" ;;
    help|-h|--help) usage ;;
    *)
        usage
        exit 2
        ;;
esac
