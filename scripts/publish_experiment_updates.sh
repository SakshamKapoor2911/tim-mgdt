#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/skapoor/tim-mgdt}"
BRANCH="${BRANCH:-master}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-chore: publish experiment updates}"

cd "$REPO"

echo "=== Publishing compact experiment updates at $(date) ==="

echo "Rebuilding Phase 19 summaries and dashboard artifacts..."
python scripts/summarize_phase19.py
python scripts/analyze_phase19_samples.py --output experiments/phase19/gsm8k_sample_analysis.json
MPLBACKEND=Agg python -c "from scripts.generate_plots import generate_phase19_reasoning_plot; generate_phase19_reasoning_plot()"
python scripts/build_dashboard.py
python scripts/phase19_readiness.py

shopt -s nullglob
paths=(
    dashboard/index.html
    dashboard/state.json
    dashboard/figures/phase19_reasoning_dissociation.png
    paper/figures/phase19_reasoning_dissociation.png
    paper/phase19_efficient_reasoning_draft.tex
    paper/phase19_workshop_outline.txt
    experiments/phase19/Mistral-7B-v0.1_mask_pruning_s*.json
    experiments/phase19/gsm8k_eval_*.json
    experiments/phase19/gsm8k_sample_analysis.json
    experiments/phase20_mask_guided/Mistral-7B-v0.1_mask_guided_width_*.json
    experiments/phase20_mask_guided/gsm8k_eval_mask_guided_*.json
    experiments/phase24_grouped_ablation/Mistral-7B-v0.1_grouped_mask_*.json
    experiments/phase24_grouped_ablation/*/gsm8k_eval_*.json
    experiments/phase24_grouped_ablation/phase24_summary_*.json
    experiments/phase24_grouped_ablation/phase24_summary_*.md
)

if ((${#paths[@]} == 0)); then
    echo "No candidate artifact paths found."
    exit 0
fi

git add "${paths[@]}"

if git diff --cached --quiet; then
    echo "No compact artifact changes to publish."
    exit 0
fi

git commit -m "$COMMIT_MESSAGE"

echo "Pushing to origin/$BRANCH..."
if ! git pull --rebase origin "$BRANCH"; then
    echo "Rebase failed. Local commit is preserved; resolve manually before pushing." >&2
    exit 1
fi

git push origin "HEAD:$BRANCH"
echo "Published compact experiment updates."
