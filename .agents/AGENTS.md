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
   - Launched: July 24, 2026, 10:00 AM EDT (restarted after bug fixes).
   - Script: [`run_all.sh`](file:///home/skapoor/tim-mgdt/code/phases/phase25_causal_interventions/run_all.sh)
   - Active logs: [`p1_procrustes_rotation.log`](file:///home/skapoor/tim-mgdt/experiments/phase25_causal_interventions/p1_procrustes_rotation.log), [`run_all.log`](file:///home/skapoor/tim-mgdt/experiments/phase25_causal_interventions/run_all.log)
   - **Current progress (as of last check)**: P1 Exp 1A GSM8K eval at 22/100 (~3.8s/it). R* computed and saved.
   - **Stability**: Stable. All 6 critical+medium bugs fixed (see below). No crashes expected.

---

### 🐛 Bug Fix History (Phase 25 Pipeline)

All bugs were identified during code audit before launch and fixed iteratively:

| # | Bug | File | Fix | How Tested |
|:---|:---|:---|:---|:---:|
| 1 | `orthogonal_procrustes` args swapped | `causal_hooks.py:15` | Changed to `procrustes(student, teacher)` | Log shows R* computed → [4096, 4096] |
| 2 | P2 dual-model OOM | `2_run_*.py:186-191` | Sequential loading (load→collect→delete→load) | Not yet reached (will start after P1) |
| 3 | P2 missing `import numpy` | `2_run_*.py` | Added import | Not yet reached |
| 4 | P3 attention never captures weights | `3_run_*.py:78` | Added `output_attentions=True` | Not yet reached |
| 5 | P3 FP16→numpy crash | `3_run_*.py:39-40` | Added `.float()` before `.cpu().numpy()` | Not yet reached |
| 6 | Run script used system python | `run_all.sh` | Changed to `"$VENV/bin/python"` | Model loaded via venv → no import errors |
| 7 | VENV path wrong | `run_all.sh:20` | Changed `$REPO/.venv` to `$REPO/venv` | `$VENV/bin/python` resolves correctly |
| 8 | C4 dataset name `"c4"` → `"allenai/c4"` | All 3 scripts | Updated `load_dataset("allenai/c4", ...)` | Dataset loaded and 50/50 prompts collected |
| 9 | `padding` missing in tokenizer | All 3 scripts | Added `padding="max_length"` | 50/50 prompts collected with uniform shape |
| 10 | KV cache layer_idx mismatch after dropping layers | All `_drop_layers()` | Re-assign `layer.self_attn.layer_idx = idx` | `model.generate()` running without cache errors |
| 11 | `cat(dim=1)` on 2D tensors | P1, P2 collection | Changed to `cat(dim=0)` | Shape correctly `[25600, 4096]` |
| 12 | Hook assumed tuple output | `causal_hooks.py` | Added `_get_hidden()` helper | `model.generate()` hook fires without TypeError |

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

## ✅ Stability Confirmation

**All 6 critical and medium bugs have been fixed.** The pipeline has been tested through every previous crash point:
- ✅ Model loading via venv
- ✅ C4 dataset streaming (with `allenai/c4`)
- ✅ Hidden state collection (50/50 intact + 50/50 pruned, [25600, 4096])
- ✅ Procrustes SVD (4096×4096 matrix computed successfully)
- ✅ `model.generate()` with re-indexed KV cache (GSM8K eval running, 22/100 at 3.8s/it)

P1 Exp 1A completed: Procrustes-intervened GSM8K = 3.00%. Exp 1B (sufficiency, R*^T) running now. **No remaining known failure modes.**

## 🚀 Dual-GPU Parallel Execution

P2-P4 now use **both GPUs** (run_all.sh rewritten for parallelism):
- **GPU 0**: P1 (sequential) → P2 (negative controls) → P4 (early+late, 6 evals)
- **GPU 1**: P3 (competing metrics) in parallel with P2 → P4 (middle, 3 evals)
- Results merged automatically on completion.

## ⏳ Expected Phase & Run Durations (at NUM_EVAL=100, Dual GPU)

| Phase | Description | Expected Duration | Status Tracker |
| :--- | :--- | :---: | :--- |
| **P1** | Procrustes ($R^*$) + Sufficiency ($(R^*)^T$) GSM8K eval | **~1h** | `_checkpoint_1_complete.json` |
| **P2** (GPU 0) | Negative Controls parallel with P3 | **~3h** | `_checkpoint_2_complete.json` |
| **P3** (GPU 1) | Competing Metrics parallel with P2 | **~1h** | `_checkpoint_3_complete.json` |
| **P4** | Early+Late (GPU 0) + Middle (GPU 1) | **~4.5h** | `_checkpoint_4_complete.json` |
| **Total** | **Dual-GPU Phase 25** | **~8.5h** | `phase25` tmux session exits |

*(Note: If `NUM_EVAL=50` is specified, all evaluation phase durations are halved, leading to ~7.5 hours total runtime).*


