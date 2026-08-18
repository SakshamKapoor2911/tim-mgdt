# Workspace Agent Instructions & Project Context

## Project Overview
- **Name**: The Fluency--Reasoning Dissociation in Depth-Pruned LLMs: A Representational Geometry Perspective
- **Target Venue**: NeurReps 2026: Workshop on Symmetry and Geometry in Neural Representations (**co-located with NeurIPS 2026**, Sydney) — **Proceedings Track, 9 pages excl. references + appendices** (PMLR archival; confirmed from the NeurReps 2026 CFP at neurreps.org). Deadline Aug 24, 2026 AoE. Double-blind via OpenReview.
- **Template**: official `jmlr.cls`/`jmlrutils.sty` (byte-identical to the official `NeurReps_2026_Template.zip`). Document class: `\documentclass[mlmain,onecolumn]{jmlr}` (per official INSTRUCTIONS for the proceedings track).
- **Execution Plan**: [`Plan.md`](file:///home/skapoor/tim-mgdt/Plan.md)
- **Paper Draft**: [`paper/main.tex`](file:///home/skapoor/tim-mgdt/paper/main.tex)
- **Overleaf Package**: [`paper/neurreps2026_overleaf.zip`](file:///home/skapoor/tim-mgdt/paper/neurreps2026_overleaf.zip) (canonical; stale `neurips2026_neurreps_overleaf.zip` retired)
- **Collaboration Plan**: [`docs/COLLABORATION.md`](file:///home/skapoor/tim-mgdt/docs/COLLABORATION.md)
- **Future ICLR 2027 Roadmap**: [`docs/FUTURE_EXPANSION_ICLR2027.md`](file:///home/skapoor/tim-mgdt/docs/FUTURE_EXPANSION_ICLR2027.md)

---

## Compile Status (IMPORTANT — verified Aug 18 2026)
- LaTeX available via **tectonic** at `~/.local/bin/tectonic` (installed user-level; no system TeX).
- `main.tex` **compiles cleanly** (zero overfull warnings) via: `cd paper && ~/.local/bin/tectonic main.tex`
- **Current compiled length: 13 pages total; main text strictly 8.0 pages** (ending on Page 8 with Section 7 Conclusion; references begin on Page 8/9; appendices on Pages 11--13). Fully compliant with the 9-page NeurReps 2026 proceedings limit.
- After any edit to `main.tex`, recompile with tectonic and re-check page count + overfull warnings.

## Experiment Status
- **Phase 25 is 100% COMPLETE.** No more GPU runs needed. All headline numbers are backed by real JSON artifacts.
- Latest integrity fixes (Aug 12–13): removed fabricated BOS attention-sink claim (synthetic figure + contradictory data); replaced fabricated BBH rotation-curve values with real lm_eval measurements (`exp_rotation_curve_bbh.json`); removed fabricated `exp_rotation_curve_perplexity.json` (PPL columns dropped from Table 3); reframed BBH as benchmark-robust dissociation corroboration.
- **PC1 framing**: directionally preserved (acute ≈0.48° after sign synchronization), with architecture-dependent energy collapse (Mistral 4.21%±0.39, Llama 0.87%, Gemma 40.21%). NOT "inverted" (the 179.52° was an SVD sign artifact).

---

## ✅ Phase 25 Complete
All 5 experiments finished. All GPU resources freed. Experiment set closed — do not launch new runs.
