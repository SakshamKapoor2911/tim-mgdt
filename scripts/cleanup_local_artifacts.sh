#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
APPLY=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/cleanup_local_artifacts.sh [--dry-run|--apply]

Removes local-only clutter that is ignored by git:
  - Python cache directories outside virtualenvs
  - .pytest_cache
  - obsolete GPT-2-era scratch folders

It intentionally does not remove checkpoints, virtualenvs, active experiment
outputs, Phase 19/20 artifacts, or git history.
EOF
}

while (($#)); do
    case "$1" in
        --apply) APPLY=1 ;;
        --dry-run) APPLY=0 ;;
        help|-h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
    shift
done

cd "$REPO"

legacy_gpt2_paths=(
    "experiments/phase2/gpt2"
    "experiments/phase5/gpt2"
    "experiments/phase6/gpt2"
    "code/phases/phase5"
    "code/phases/phase5_1"
    "code/phases/phase8"
    "code/phases/phase10"
    "code/interventions/dynamic_early_exit"
    "code/interventions/inattention_conversion"
    "code/interventions/kl_divergence_proxy"
    "code/interventions/low_rank_compression"
    "code/interventions/quantization_maps"
)

paths=()

while IFS= read -r cache_dir; do
    paths+=("$cache_dir")
done < <(
    find . \
        -path './.git' -prune -o \
        -path './venv' -prune -o \
        -path './venv_eval' -prune -o \
        -name '__pycache__' -type d -print
)

[[ -d ".pytest_cache" ]] && paths+=(".pytest_cache")

for path in "${legacy_gpt2_paths[@]}"; do
    [[ -e "$path" ]] && paths+=("$path")
done

if ((${#paths[@]} == 0)); then
    echo "No local cleanup candidates found."
    exit 0
fi

echo "Local cleanup candidates:"
printf '  %s\n' "${paths[@]}"

if ((APPLY == 0)); then
    echo
    echo "Dry run only. Re-run with --apply to remove these paths."
    exit 0
fi

for path in "${paths[@]}"; do
    rm -rf -- "$path"
done

echo "Removed ${#paths[@]} local cleanup candidates."
