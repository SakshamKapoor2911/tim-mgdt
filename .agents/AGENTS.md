# Workspace Agent Instructions & Project Context

## Project Overview
- **Name**: Suturing the Cut: Subspace Rotation and the Fluency--Reasoning Dissociation in Incision Boundary Healing of Depth-Pruned LLMs
- **Target Venue**: BlackboxNLP 2026: The 9th Workshop on Analyzing and Interpreting Neural Networks for NLP (**co-located with EMNLP 2026 in Budapest, Hungary** — *Full Archival Track*, Up to 8 Pages)
- **Submission Deadline**: July 30, 2026 (AOE)
- **Review Status**: **Top-Tier Workshop Paper (9.0/10)** — 90%--95% Acceptance Probability (Special Track Oral Recommendation)  
- **Manifest**: [`repo_map.yaml`](file:///home/skapoor/tim-mgdt/repo_map.yaml)
- **Execution Plan**: [`Plan.md`](file:///home/skapoor/tim-mgdt/Plan.md)
- **Paper Draft**: [`paper/main.tex`](file:///home/skapoor/tim-mgdt/paper/main.tex)
- **Overleaf Package**: [`paper/blackboxnlp2026_overleaf.zip`](file:///home/skapoor/tim-mgdt/paper/blackboxnlp2026_overleaf.zip)
- **Review Log**: [`main_BlackboxNLP2026_review.txt`](file:///home/skapoor/tim-mgdt/main_BlackboxNLP2026_review.txt)
- **Collaboration Plan**: [`docs/COLLABORATION.md`](file:///home/skapoor/tim-mgdt/docs/COLLABORATION.md)
- **Future ICLR 2027 Roadmap**: [`docs/FUTURE_EXPANSION_ICLR2027.md`](file:///home/skapoor/tim-mgdt/docs/FUTURE_EXPANSION_ICLR2027.md)

---

## Active Paper Execution Status & Next Steps

1. **GPU 0 & GPU 1 Compute Pipeline**: **100% COMPLETED** for initial baseline.
2. **Phase 25 Causal Interventions Pipeline**: **RUNNING** in background `tmux` session `phase25`.
   - Launched: July 24, 2026.
   - Script: [`run_all.sh`](file:///home/skapoor/tim-mgdt/code/phases/phase25_causal_interventions/run_all.sh)
   - Active logs: [`p1_procrustes_rotation.log`](file:///home/skapoor/tim-mgdt/experiments/phase25_causal_interventions/p1_procrustes_rotation.log), [`run_all.log`](file:///home/skapoor/tim-mgdt/experiments/phase25_causal_interventions/run_all.log)

---

## 🛑 Critical Monitoring Guidelines (Read-Only Safety)

An active pipeline run is executing in `tmux` session `phase25` on GPU 0. To prevent interrupting, corrupting, or duplicating runs:
- **Do NOT run modifying terminal commands** or python execution commands (e.g. `run_all.sh`, `1_run_*.py`, etc.).
- **Do NOT launch new tmux sessions** named `phase25` or kill the active `phase25` session.
- **Monitoring ONLY**: To check execution progress, use strictly read-only checks:
  1. Capture current shell view: `tmux capture-pane -t phase25 -p`
  2. View log file tail: `tail -n 50 experiments/phase25_causal_interventions/run_all.log`
  3. List completed checkpoints: `ls -la experiments/phase25_causal_interventions/` (look for `_checkpoint_X_complete.json`)

---

## ⏳ Expected Phase & Run Durations (at NUM_EVAL=100)

| Phase | Description | Expected Duration | Status Tracker |
| :--- | :--- | :---: | :--- |
| **P1** | Procrustes Alignment ($R^*$) + Sufficiency ($(R^*)^T$) | **~2 hours** | `_checkpoint_1_complete.json` |
| **P2** | Negative Controls (Placebo, Wrong Layer, Wrong PC) | **~3 hours** | `_checkpoint_2_complete.json` |
| **P3** | Fixed-Depth Competing Metrics Correlation Matrix | **~1 hour** | `_checkpoint_3_complete.json` |
| **P4** | Location Sweep (Early/Middle/Late $\times$ 3 seeds) | **~9 hours** | `_checkpoint_4_complete.json` |
| **Total** | **Complete Phase 25 Evaluation Pipeline** | **~15 hours** | `phase25` tmux session exits |

*(Note: If `NUM_EVAL=50` is specified, all evaluation phase durations are halved, leading to ~7.5 hours total runtime).*


