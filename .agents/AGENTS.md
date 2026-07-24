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
- **Phase 25 Results Analysis**: [`docs/analysis/phase25_results_analysis.md`](file:///home/skapoor/tim-mgdt/docs/analysis/phase25_results_analysis.md)

---

## Active Paper Execution Status & Next Steps

**Phase 25 is 100% COMPLETE.** All experiments (P1–P5) have finished. No more GPU runs needed.

- **P1**: Sufficiency test (34% → 10%, strong causal evidence) ✅
- **P2**: Negative controls (random=0%, wrong-layer=0-3%, wrong-PC=0-3%) ✅
- **P3**: Competing metrics (θ unique vs scalar metrics, μ/σ drift p>0.4) ✅
- **P4**: Location sweep (0%/2%/4.33% across layers, n=300) ✅
- **P5**: PC₁ un-rotation (θ=179.52°, projection=−1.000, GSM8K=4.00%) ✅

**Key finding**: PC₁ is not rotated by 29.34° — it is **inverted** (179.52°). The 29.34° figure is the mean across top-10 PCs, masking this anti-alignment. Recovery experiments all converge to 0–4% noise floor — inference-time geometric correction is insufficient because downstream weights are calibrated on the inverted PC₁.

**Next step**: Paper reframing per `docs/analysis/phase25_results_analysis.md` §9. The experiment set is closed.

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
| 13 | P3 CPU SVD FP16 crash | `3_run_*.py:94` | Cast hidden state to `.float()` before CPU SVD | P3 runs CPU SVD successfully without LAPACK error |
| 14 | P3 SDPA ignores attention hooks | `3_run_*.py` model load | Set `attn_implementation="eager"` to materialize matrix | Hook retrieves attention weights, entropy calculated |
| 15 | P3 attention entropy NaN underflow | `3_run_*.py:90` | Cast weights to `.float()` before clamp(min=1e-10) | `exp_3_competing_metrics.json` has non-NaN values |

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

**All 14 critical and medium bugs have been fixed.** The pipeline has successfully completed:
- ✅ Model loading via venv and streaming on C4 (`allenai/c4`)
- ✅ Hidden state collection and Procrustes alignment ($R^*$)
- ✅ P1 Exp 1A (GSM8K realigned accuracy = 3.00%) & Exp 1B (Sufficiency baseline = 34.00%)
- ✅ P3 fixed-depth correlation metrics (computed eager attention and effective ranks at 10:44:17 AM EDT)
- ✅ Currently running P2 on GPU 0 without crashes

**Phase 25 100% complete (all P1–P5). Experiment set closed.**

## 🚀 Dual-GPU Parallel Execution

P2-P4 now use **both GPUs** (run_all.sh rewritten for parallelism):
- **GPU 0**: P1 (sequential) → P2 (negative controls) → P4 (early+late, 6 evals)
- **GPU 1**: P3 (competing metrics) in parallel with P2 → P4 (middle, 3 evals)
- Results merged automatically on completion.

## ✅ Phase 25 Complete

All 5 experiments finished. All GPU resources freed. Total runtime: ~3 hours (dual-GPU), ~6 hours wall-clock.
