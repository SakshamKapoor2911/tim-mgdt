# Reviewer-Proofing Execution Plan: Suturing the Cut (Subspace Rotation & Fluency--Reasoning Dissociation)

**Master Framing**: Do NOT simply ask *"Make the paper stronger."* Ask: **"Design experiments that eliminate the strongest reviewer objections."**  
**Core Goal**: Construct an unassailable causal chain by systematically ruling out alternative explanations and proving causal sufficiency/necessity.  
**Baseline Branch**: `blackboxnlp-2026-baseline` (Commit `2401589`)  
**Active Development Branch**: `feature/subspace-suturing-improvements`  

---

## 🎯 Logical Flow of the Final Paper

The revised manuscript should read like a formal proof:
1. **Observation**: Depth pruning consistently induces a localized representational rotation at the incision boundary.
2. **Predictive Dominance**: Subspace rotation angle predicts downstream reasoning collapse significantly better than competing scalar metrics (residual norm, LayerNorm drift, attention entropy, effective rank).
3. **Necessity**: Targeted ablation of $\text{PC}_1$ (the primary eigenvector subjected to rotation) destroys reasoning accuracy while preserving fluency.
4. **Sufficiency**: 
   - **Induction**: Artificially rotating hidden states in an intact model without pruning destroys reasoning (rotation is *sufficient* to cause failure).
   - **Recovery**: Direct Orthogonal Procrustes subspace realignment ($R^* \in \text{SO}(d)$) or targeted $\text{PC}_1$ un-rotation in the pruned model restores reasoning (rotation repair is *sufficient* for recovery).
5. **PEFT Mechanism**: Boundary LoRA restores local BOS attention sink mass and fluency without realigning the rotated subspace, explaining the fluency-reasoning dissociation.
6. **Alternative Explanations Eliminated**: Explicit negative controls prove that non-rotational factors fail to account for reasoning collapse.

---

## 📋 Priority Matrix & Actionable Roadmap

| Priority | Focus Area | Key Deliverable | Reviewer Objection Eliminated | Estimated ROI |
| :--- | :--- | :--- | :--- | :---: |
| **Priority 0** | **Claim Calibration** | Precision text pass across `paper/main.tex` | "Overclaiming / Unsubstantiated causality statements" | ★★★★★ |
| **Priority 1** | **Causal Interventions** | Procrustes un-rotation ($R^*$) & Artificial rotation | "Correlation is not causation / How do you know rotation causes collapse?" | ★★★★★★★★★★ |
| **Priority 2** | **Sufficiency vs Necessity** | Inject repaired $\text{PC}_1$ into pruned model | "Ablation only proves necessity, not sufficiency" | ★★★★★ |
| **Priority 3** | **Alternative Explanations** | Metric correlation table (Rotation vs Norm vs Entropy) | "What if it's just LayerNorm drift, attention collapse, or norm explosion?" | ★★★★★ |
| **Priority 4** | **Robustness & Locations** | Multi-seed + Early/Mid/Late layer drop evaluation | "Is this an artifact of picking middle layers or specific random seeds?" | ★★★★☆ |
| **Priority 5** | **PC Spectrum Ranking** | Spectrum ablation across $\text{PC}_1 \dots \text{PC}_{20}$ | "Why $\text{PC}_1$? What about the rest of the principal spectrum?" | ★★★★☆ |
| **Priority 6** | **Architecture Expansion** | Qwen-2.5-7B validation run | "Does this hold beyond Llama/Mistral/Gemma?" | ★★★☆☆ |
| **Priority 7** | **Benchmark Generalization** | BBH / StrategyQA multi-step evaluation | "GSM8K is overfitted / Single benchmark risk" | ★★★★☆ |
| **Priority 8** | **Geometric Sensitivity** | CKA, SVCCA, PWCCA & top-$k$ principal angle sweeps | "Sensitivity to metric choice or top-10 cutoff" | ★★★★☆ |
| **Priority 9** | **Mechanistic Visualizations**| Publication scatter plots & PC trajectory figures | "Clarity of mechanistic proof" | ★★★☆☆ |
| **Priority 10**| **Negative Controls** | Random rotation, wrong layer/PC repair controls | "Placebo / Artifact controls missing" | ★★★★★ |

---

## 🔍 Detailed Priority Specifications

### Priority 0: Claim Calibration & Precision Pass (Mandatory)
* **Goal**: Categorize every claim in `paper/main.tex` into (A) Direct Experimental Proof, (B) Indirect Evidence, (C) Working Hypothesis.
* **Target Changes**:
  - Replace *"We prove $\text{PC}_1$ is causally responsible..."* $\to$ *"We provide direct causal intervention evidence that $\text{PC}_1$ realignment..."*
  - Replace *"universal rotational shift"* $\to$ *"consistent rotational shift across evaluated transformer architectures"*

### Priority 1: Direct Causal Intervention (Highest ROI ★★★★★★★★★★)
* **Experiment 1A: Orthogonal Procrustes Realignment ($R^*$)**:
  - Compute $R^* = U \det(U V^T) V^T \in \text{SO}(d)$ via SVD of student-teacher cross-covariance at the incision boundary.
  - Apply $h' = R^* h$ during forward pass and evaluate GSM8K/BBH reasoning.
* **Experiment 1D: Artificial Rotation in Intact Model**:
  - Apply synthetic rotation $R_{\theta}$ to intact hidden states without dropping layers.
  - Verify if rotation alone is *sufficient* to destroy multi-step reasoning.

### Priority 2: Sufficiency vs. Necessity
* **Goal**: Inject un-rotated/repaired $\text{PC}_1$ back into pruned model to test if reasoning is restored.

### Priority 3: Ruling Out Alternative Explanations
* **Goal**: Compute a unified correlation matrix comparing:
  1. Subspace rotation angle $\theta$
  2. Residual vector norm $\|h\|_2$
  3. LayerNorm mean/variance drift $(\mu, \sigma)$
  4. Attention entropy $H(A)$
  5. Effective rank / Singular value decay
* **Hypothesis**: Subspace rotation angle $\theta$ will exhibit the highest correlation ($r_s > 0.85$) with GSM8K accuracy degradation.

### Priority 4: Robustness & Pruning Location Sweeps
* **Goal**: Run Early (layers 3-6), Middle (layers 11-14), Late (layers 20-23) block drops across 3 random seeds with 95% Clopper-Pearson / Bootstrap CIs.

### Priority 5: Complete Principal Component Spectrum Ranking
* **Goal**: Rank and ablate $\text{PC}_1, \text{PC}_2, \dots, \text{PC}_{20}$ individually to map the exact causal energy landscape.

### Priority 6-10: Generalization, Geometry Sensitivity & Controls
* **Geometry Sensitivity**: Compare Principal Angles, CKA, SVCCA, and PWCCA across top-5 to top-50 components.
* **Negative Controls**: Test random orthogonal matrix rotation $R_{\text{rand}}$, repair at unperturbed layers, and random direction injection to confirm specificity.

---

## 🛑 Strict Execution Guardrails ("Things NOT to Do")

1. **Do NOT add 10 arbitrary benchmarks**: Focus on 1-2 decisive reasoning tasks (GSM8K, BBH) with deep causal analysis.
2. **Do NOT run endless LoRA hyperparameter sweeps**: Ranks $r=16, 128$ are already established.
3. **Do NOT make bigger claims**: Make tighter, bulletproof claims with unassailable empirical backing.
4. **Do NOT report uncorrelated observations**: Every figure/table must directly answer a Reviewer #2 objection.
