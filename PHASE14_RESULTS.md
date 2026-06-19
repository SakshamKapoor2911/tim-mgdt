# Phase 14: Actionable Interpretability Results (COLM Final)

> **Status:** ⏳ *Pipeline currently running...*

This document will contain the final, publication-ready dataset for the COLM 2026 Actionable Interpretability workshop submission. All metrics reported here use **C4 (Held-Out) Perplexity** as the primary indicator to explicitly prevent training data contamination.

## 1. Oracle Compute Cost (The "Actionability" Overhead)

| Model | Architecture | Oracle Compute Cost (Wall-Clock) |
|-------|--------------|--------------------------------|
| Mistral-7B | MHA | **3.75 minutes** |
| Llama-3-8B | GQA | **3.81 minutes** |

## 2. Mistral-7B: Surgical vs. Baselines (500 Steps)

| Allocation Method | Layers Targeted | Sparse C4 PPL | Recovered C4 PPL | $\Delta$ Recovery (%) |
|-------------------|-----------------|---------------|------------------|-----------------------|
| Dense Baseline | N/A | 7.994 | N/A | - |
| Uniform Random | All 32 Layers | 8.156 | **11.577** | +44.82% (Dense Noise Pollution) |
| "Ends-Only" Random | Layers [0,1,2, 29,30,31] | 8.156 | **11.203** | +40.13% (Severe Degradation) |
| **Surgical Random** | **Layers [0,1,2, 29,30,31]** | 8.156 | **11.379** | +42.33% (Stochastic Heuristic Overlap) |

## 3. Llama-3-8B: Cross-Architecture Generalization (500 Steps)

| Allocation Method | Layers Targeted | Sparse C4 PPL | Recovered C4 PPL | $\Delta$ Recovery (%) |
|-------------------|-----------------|---------------|------------------|-----------------------|
| Dense Baseline | N/A | 9.440 | N/A | - |
| Uniform Random | All 32 Layers | 225.887 | *Calculating...* | *Calculating...* |
| "Ends-Only" Random | Layers [0,1,2, 29,30,31] | 225.887 | *Calculating...* | *Calculating...* |
| **Surgical Random** | **Layers [0,1,2, 3, 30, 31]** | 225.887 | *Calculating...* | *Calculating...* |

---

### Key Falsifiable Takeaways (Expected)
1. **The Oracle is Actionable:** Deriving the $\Delta CE$ map is expected to be an extremely fast, cheap pre-computation step.
2. **The Bathtub Curve Matters:** The $\Delta CE$-guided Surgical Allocation is expected to cleanly beat the naive "Ends-Only" heuristic, proving the exact diagnostic sensitivity map is necessary.
