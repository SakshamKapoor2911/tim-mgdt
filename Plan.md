# Reviewer-Proofing Master Execution Plan: Suturing the Cut

**Master Framing**: Do NOT simply ask *"Make the paper stronger."* Ask: **"Design experiments that eliminate the strongest reviewer objections."**  
**Core Target**: ARR / BlackboxNLP 2026 Archival Commitment (**August 28, 2026** — 5-Week Horizon).  
**Immediate Action**: Priority 0 Claim Calibration in `paper/main.tex` (Today, July 24, 2026).  
**Baseline Branch**: `blackboxnlp-2026-baseline` (Commit `2401589`)  
**Active Development Branch**: `feature/subspace-suturing-improvements`  

---

## 🔍 Section 1: Situational & Paper Gap Audit

| Paper Claim / Section | TeX Line Ref | Claimed in Paper? | Code / Results Exist? | Gap & Required Action |
| :--- | :---: | :---: | :---: | :--- |
| **Claim Calibration Pass** | Lines 43–45, 220, 321 | Yes ("We prove", "causally responsible") | Partial | **P0 (Immediate)**: Soften causal claims to reflect empirical scope until P1 completes. |
| **Direct PC Realignment** | Line 220–221 | Yes | **0%** | **P1 (MVP)**: Paper asserts PC1 projection realignment, but code does not exist. Must build Procrustes & PC1 repair hooks. |
| **Orthogonal Procrustes $R = UV^T$** | Line 309 | Yes (Future outlook) | **0%** | **P1 (MVP)**: Implement $R^* = U \det(UV^T)V^T$ forward hook intervention and test GSM8K/BBH recovery. |
| **Artificial Rotation in Intact Model** | Abstract (line 43–44) | Partial | **0%** | **P1 (MVP)**: Implement synthetic rotation $R_\theta$ in intact model to prove rotation alone is *sufficient* to destroy reasoning. |
| **Negative Controls (Placebo / Wrong Layer)** | Section 5 (lines 267–293) | Partial (9 baselines exist) | **20%** | **P2 (Promoted)**: Add placebo random rotation, wrong-layer repair, wrong-PC repair to satisfy Reproducibility Track reviewers. |
| **Competing Scalar Metrics (Fixed Depth)** | Section 4.4 / 5.1 | Partial ($r_s=1.0$ across depth) | **40%** | **P3**: Measure Rotation vs Norm vs LayerNorm Drift vs Attention Entropy at **fixed skip-depth** ($k=4$). |
| **Pruning Location Sweeps** | Section 4 | Partial (Middle layers 11-14) | **60%** | **P4**: Run Early (3-6), Middle (11-14), Late (20-23) block drops across 3 seeds with 95% CIs. |
| **Full Spectrum Ranking ($\text{PC}_1 \dots \text{PC}_{20}$)** | Table 3 (lines 222–240) | Partial ($\text{PC}_1, \text{PC}_2, \text{PC}_8$) | **80%** | **P5 (Demoted)**: Low marginal value (lines 230-237 already establish $\text{PC}_1$ dominance). |

---

## 📅 Section 2: Deadline Anchoring & 5-Week Milestone Roadmap

- **Reproducibility Track Target (July 24, 2026 — Today)**: Execute **P0 Claim Calibration** in `paper/main.tex` to ensure zero over-claiming in current manuscript state.
- **ARR Commitment Target (August 28, 2026 — 5-Week Horizon)**: Complete P1 through P4 for final camera-ready submission.

```mermaid
gantt
    title ARR Commitment Execution Timeline (July 24 - August 28, 2026)
    dateFormat  YYYY-MM-DD
    section Phase 0: Calibration
    P0 Text Pass & Compile       :active, p0, 2026-07-24, 2d
    section Phase 1: Causal Intervention
    P1 Scaffolding & Coding       :p1_code, 2026-07-26, 3d
    P1 Procrustes & Rotation Eval :p1_eval, 2026-07-29, 4d
    section Phase 2: Negative Controls
    P2 Placebo & Control Runs     :p2_eval, 2026-08-02, 5d
    section Phase 3: Metrics Matrix
    P3 Fixed-Depth Metrics Sweep  :p3_eval, 2026-08-07, 5d
    section Phase 4: Robustness
    P4 Multi-Location & Seeds     :p4_eval, 2026-08-12, 6d
    section Paper Polish & Freeze
    Paper Integration & PDF Freeze:paper, 2026-08-18, 10d
```

---

## 🚀 Section 3: Reprioritized Hierarchy, Resource Budget & Dependencies

### Submission MVP (Mandatory for ARR / BlackboxNLP Archival)

| Priority | Priority Name | Resource Budget | Dependency | Primary Reviewer Objection Eliminated | Estimated ROI |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **P0** | **Immediate Claim Calibration** | 2 coding hrs / 0 GPU hrs | None | "Overclaiming / Unsubstantiated causality statements" | ★★★★★ |
| **P1** | **Direct Causal Interventions** | 6 coding hrs / 12 GPU hrs | P0 | "Correlation is not causation / How do you know rotation causes collapse?" | ★★★★★★★★★★ |
| **P2** | **Mechanism Negative Controls** | 4 coding hrs / 8 GPU hrs | P1 | "Placebo / Artifact controls missing (random rotation, wrong-layer repair)" | ★★★★★ |
| **P3** | **Fixed-Depth Competing Metrics** | 4 coding hrs / 6 GPU hrs | P1 | "What if it's just scalar norm drift, LayerNorm, or attention entropy at fixed depth?" | ★★★★★ |
| **P4** | **Multi-Location & Seed Robustness**| 3 coding hrs / 16 GPU hrs | P1 | "Is this an artifact of picking middle layers or specific random seeds?" | ★★★★☆ |

### Optional Expansion (Nice-to-Have if time permits)

| Priority | Priority Name | Resource Budget | Dependency | Reason for Lower Rank | Estimated ROI |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **P5** | **Full Spectrum Sweep ($\text{PC}_{1-20}$)**| 2 coding hrs / 4 GPU hrs | P1 | Low marginal value; Table 3 already proves $\text{PC}_1$ dominates (23% drop vs 0% for $\text{PC}_2$). | ★★★☆☆ |
| **P6** | **Architecture Expansion (Qwen)** | 4 coding hrs / 10 GPU hrs | P1 | Mistral, Llama-3, Gemma-2 already cover global & SWA architectures. | ★★★☆☆ |
| **P7** | **Benchmark Expansion (StrategyQA)**| 3 coding hrs / 8 GPU hrs | P1 | GSM8K and BBH multi-step arithmetic already establish task generalization. | ★★★★☆ |
| **P8** | **Geometric Metrics (CKA/SVCCA)** | 4 coding hrs / 4 GPU hrs | P1 | SVD principal angles already provide exact rotational degrees. | ★★★★☆ |

---

## 🏗️ Section 4: Technical Scaffolding for Priority 1 (Procrustes & Artificial Rotation)

### Implementation Specification

1. **Module**: `code/phases/phase25_causal_interventions/causal_hooks.py`
2. **Dependencies**: `torch`, `scipy.linalg.orthogonal_procrustes`, `transformers`
3. **Complexity & Architecture**:
   - **Forward Hook Injection**: Intercept hidden state tensor $H \in \mathbb{R}^{B \times S \times D}$ at Layer 10 (pre-incision) and Layer 15 (post-incision).
   - **Procrustes Realignment Operator**:
     $$\min_{R \in \text{SO}(D)} \| H_{\text{student}} R - H_{\text{teacher}} \|_F^2 \implies R^* = U \det(U V^T) V^T \quad \text{where } H_{\text{teacher}}^T H_{\text{student}} = U \Sigma V^T$$
   - **Artificial Rotation Operator**:
     Apply synthetic Givens/Householder rotation matrix $R_{\theta}$ to intact model activations at Layer 10 with $\theta \in [0^\circ, 45^\circ]$.

```python
# Technical Scaffolding Pseudocode for Priority 1 Hook Injection
import torch
import numpy as np
from scipy.linalg import orthogonal_procrustes

class SubspaceInterventionHook:
    def __init__(self, mode="procrustes", R_target=None):
        self.mode = mode
        self.R_target = R_target # torch.Tensor [D, D]
        
    def __call__(self, module, inputs, output):
        # output[0] shape: [batch, seq_len, hidden_dim]
        h = output[0]
        if self.mode == "rotate" and self.R_target is not None:
            h_intervened = torch.matmul(h, self.R_target.to(h.device, dtype=h.dtype))
            return (h_intervened,) + output[1:]
        return output
```

---

## 🛑 Section 5: Strict Execution Guardrails

1. **Do NOT add 10 arbitrary benchmarks**: Focus on 1-2 decisive reasoning tasks (GSM8K, BBH) with deep causal analysis.
2. **Do NOT run endless LoRA hyperparameter sweeps**: Ranks $r=16, 128$ are already established.
3. **Do NOT make bigger claims**: Make tighter, bulletproof claims with unassailable empirical backing.
4. **Do NOT report uncorrelated observations**: Every figure/table must directly answer a Reviewer #2 objection.
