# Future Main Conference Expansion Roadmap (ICLR 2027 / ACL 2027)

## 📌 Project & Paper Metadata
- **Proposed Working Title**: *Un-Rotating the Cut: Orthogonal Procrustes Adaptation and Directional Subspace Realignment for Post-Pruning Reasoning Recovery in LLMs*
- **Target Venue**: ICLR 2027 (Oral/Spotlight) / ACL 2027 Main Conference (Full 9--12 Page Format)
- **Primary Goal**: Transition from the diagnostic finding of the **Fluency--Reasoning Dissociation** (BlackboxNLP 2026) to a **positive, state-of-the-art solution** that directly repairs post-pruning reasoning collapse without full retuning.
- **Core Hypothesis**: Physical depth pruning causes a $29.34^\circ - 37.84^\circ$ subspace rotation targeting the primary reasoning eigenvector ($\text{PC}_1$). Standard next-token fine-tuning (LoRA, MSE distillation, norm matching) fails because it optimizes scalar/likelihood magnitude while leaving $\text{PC}_1$ un-rotated ($\Delta \theta = 0.03^\circ$). An explicit **Orthogonal Procrustes PEFT (OP-LoRA)** adapter that enforces directional subspace realignment at the incision boundary will un-rotate $\text{PC}_1$ and recover multi-step reasoning capabilities.

---

## 📐 Mathematical Formulation: Orthogonal Procrustes PEFT (OP-LoRA)

### 1. Cross-Subspace Inner Product Matrix
Given intact teacher hidden states $H^{(T)}_l \in \mathbb{R}^{N \times d}$ and pruned student hidden states $H^{(S)}_l \in \mathbb{R}^{N \times d}$ at incision layer $l$, compute the uncentered cross-covariance matrix:
$$M_l = (H^{(S)}_l)^T H^{(T)}_l \in \mathbb{R}^{d \times d}$$

### 2. Singular Value Decomposition (SVD)
Compute the full or truncated rank-$r$ SVD of $M_l$:
$$\text{SVD}(M_l) = U \Sigma V^T, \quad U, V \in \mathbb{R}^{d \times d}$$

### 3. Optimal Orthogonal Alignment Matrix ($R^*$)
The closed-form solution to the classical Orthogonal Procrustes problem $\min_{R \in \text{O}(d)} \| H^{(S)}_l R - H^{(T)}_l \|_F^2$ is given by:
$$R^* = U \text{diag}\left(1, 1, \dots, 1, \det(U V^T)\right) V^T \in \text{SO}(d)$$
where the determinant correction guarantees a proper rotation without reflection.

### 4. OP-LoRA Layer Formulation
Instead of standard additive LoRA $h' = h + W_{\text{down}} W_{\text{up}} h$, OP-LoRA applies a decoupled rotational transform followed by localized low-rank boundary adaptation:
$$h'_{\text{incision}} = R^* h_{\text{incision}} + \alpha \cdot B A h_{\text{incision}}$$
where $R^* \in \text{SO}(d)$ is frozen after zero-shot calibration and $(B, A)$ are fine-tuned on task-aligned reasoning trajectories.

---

## 🔬 Key Work Packages & Detailed Tasks

### Work Package 1: Theoretical First-Principles Derivation
- [ ] **Task 1.1**: Formalize the mathematical proof linking layer removal to operator norm bounds in residual networks ($h_{l+k} = h_l + \sum_{j=1}^k f_j(h_{j-1})$). Prove that dropping $k$ non-linear block operators induces an upper-bounded angular perturbation $\theta \le \arcsin\left(\frac{\| \Delta h \|}{\| h \|}\right)$.
- [ ] **Task 1.2**: Derive the directional sensitivity spectrum of $\text{PC}_1$ vs $\text{PC}_2 \dots \text{PC}_d$ under autoregressive decoding. Prove why next-token prediction gradients on general text corpora (C4) yield near-zero projection along rotated $\text{PC}_1$.

### Work Package 2: OP-LoRA Adapter Architecture & Code Implementation
- [ ] **Task 2.1**: Implement `code/interventions/procrustes_adapter.py` providing:
  - `ProcrustesCalibrationHook`: Extracts paired hidden state matrices $H^{(T)}, H^{(S)}$ across calibration sequences.
  - `OrthogonalRotationLayer`: Implements efficient PyTorch $R^*$ matrix multiplication with FP16/BF16 stability guards and GPU kernel acceleration.
  - `OPLoRAModule`: Combines frozen orthogonal rotation matrix $R^*$ with trainable boundary LoRA adapters ($r=16, 32$).
- [ ] **Task 2.2**: Implement differentiable rotation gates ($R(\phi)$ parameterised by Lie Algebra $\mathfrak{so}(d)$ generators) to test end-to-end backpropagation through rotation angles.

### Work Package 3: Benchmark & Task Scaling (Beyond GSM8K)
- [ ] **Task 3.1 (Mathematical & Symbolic Reasoning)**:
  - Full GSM8K test set ($N=1319$ problems).
  - MATH Benchmark (5,000 challenging competition math problems across 7 sub-fields).
  - SVAMP (Robust word problem benchmark).
- [ ] **Task 3.2 (Multi-Step Algorithmic & General Reasoning)**:
  - MMLU-Pro (Reasoning-focused MMLU subset).
  - Big-Bench Hard (BBH: all 23 challenging algorithmic tasks).
- [ ] **Task 3.3 (Code Generation & Logic)**:
  - HumanEval ($N=164$ python coding tasks).
  - MBPP (Mostly Basic Python Problems).

### Work Package 4: Model & Architectural Scale Expansion
- [ ] **Task 4.1 (70B Scale Validation)**:
  - Meta-Llama-3.1-70B-Instruct under contiguous depth drops ($k=4, 8$).
  - Qwen-2.5-72B-Instruct under depth drops.
- [ ] **Task 4.2 (Advanced Backbones & MoE)**:
  - DeepSeek-R1-Distill-Llama-70B (Reasoning-distilled model architecture).
  - Mixtral-8x7B (Mixture-of-Experts contiguous expert layer drop).
  - Gemma-2-27B (Large-scale Sliding Window Attention SWA).

---

## 📅 Execution & Target Timeline

| Milestone | Target Date | Primary Deliverable |
| :--- | :--- | :--- |
| **M1: Workshop Submission** | July 30, 2026 | Submit current manuscript to BlackboxNLP 2026 @ EMNLP 2026. |
| **M2: OP-LoRA Implementation** | August 15, 2026 | Complete `code/interventions/procrustes_adapter.py` and unit tests. |
| **M3: Pilot 7B/8B OP-LoRA Run** | September 1, 2026 | Run OP-LoRA on Mistral-7B & Llama-3-8B; measure GSM8K recovery vs boundary LoRA. |
| **M4: Benchmark Scaling** | September 20, 2026 | Evaluate OP-LoRA on MATH, BBH, and HumanEval. |
| **M5: 70B Model Scaling** | October 10, 2026 | Run OP-LoRA on Llama-3.1-70B & Qwen-2.5-72B on dual GPU setup. |
| **M6: Paper Drafting** | November 1, 2026 | Synthesize 12-page manuscript for ICLR 2027 / ACL 2027. |

---

## 🗂️ Proposed File Structure for Expansion

```
code/
├── interventions/
│   ├── procrustes_adapter.py      # Orthogonal Procrustes R* matrix computation & PyTorch module
│   ├── lie_algebra_rotation.py    # Differentiable so(d) matrix exponential rotation layer
├── metrics/
│   ├── subspace_energy.py         # Exact PC1..PC10 energy projection metrics
├── phases/
│   ├── phase26_procrustes_eval/   # Experimental runner scripts for OP-LoRA
│   │   ├── run_op_lora_mistral.sh
│   │   ├── run_op_lora_llama70b.sh
│   │   ├── eval_math_full.py
│   │   └── eval_mmlu_pro.py
docs/
├── FUTURE_EXPANSION_ICLR2027.md   # This detailed roadmap
```

---

*This document serves as the authoritative blueprint for expanding the BlackboxNLP 2026 diagnostic findings into a tier-1 main conference submission.*

---

## 📊 Main-Conference Gap Analysis (Updated 2026-07-24)

### Current Paper Status
- **Workshop score**: 8.5/10 (external reviewer). 80-90% poster, 20-35% spotlight.
- **Strengths**: PC₁ inversion discovery, sufficiency test, 9-intervention falsification, cross-model evidence, N=500 CIs.
- **Limits**: Diagnostic only. No training-time recovery. 7-8B models only. GSM8K + BBH only.

### Gaps to Main Conference

| Gap | Why it matters | Effort | Priority |
|:---|:---|:---:|:---:|
| **1. Training-time PC₁ recovery** | Paper diagnoses but doesn't solve. Main conference expects a proof-of-concept fix. | **6–8 weeks** | P0 |
| **2. More reasoning benchmarks** | GSM8K + BBH is thin. GPQA, MMLU-Pro, MATH, or AIME needed for generalization. | **1–2 weeks** | P1 |
| **3. Larger model (13-70B)** | 7-8B is small for main conference. At minimum a 13B, ideally 70B. | **2–4 weeks** | P1 |
| **4. Prior recovery baselines** | Only compared against own LoRA variants. ShortGPT, LaCo, or continued pre-training needed. | **1 week** | P2 |
| **5. Theoretical grounding** | Empirical geometry claim needs formal framework for why PC₁ rotation → reasoning failure. | **2–3 weeks** | P2 |

### Effort by Venue Tier

| Venue tier | What's needed | Effort |
|:---|:---|:---:|
| **Mid-tier** (COLM, ACL Findings, EMNLP) | Gaps #2 + #4 + maybe a 13B run from #3 | **2–3 weeks** |
| **Top-tier** (NeurIPS, ICLR, ICML) | All 5 gaps addressed | **6–8 weeks** |

### Critical Path
Training-time PC₁ recovery (Gap #1) is the bottleneck. It's the highest-risk, highest-reward experiment. If it succeeds: "we diagnosed AND fixed the problem" → strong NeurIPS/ICLR paper. If it fails: the diagnostic story alone can still reach mid-tier venues with Gaps #2 and #4.

### Recommended Strategy
**Submit to workshops now** (diagnostic paper, 8.5/10 ready). In parallel, begin training-time PC₁ recovery experiments. If successful by Oct-Nov, target ICLR 2027 main conference with the recovery as the headline result.
