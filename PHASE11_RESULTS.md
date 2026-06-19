# Phase 11 Results

This document will contain the final analysis, outputs, and metrics from the parallel Phase 11 execution of Mistral-7B and Llama-3-8B.

*Status: Jobs currently running on GPU 0 and GPU 1. Results will be populated here once complete.*

## Mistral-7B (GPU 1)
*   **Stage 1 (Structured Pruning):** *Pending*
*   **Stage 2 (IMP Search):** *Pending*
*   **Stage 3 (Causal Patching):** *Pending*
*   **Stage 4 (Activation Healing):** *Pending*

## Llama-3-8B (GPU 0)
*   **Stage 1 & 2 (IMP Search):** *Success*. Reached **9.3% global sparsity** with only a **+1.75% Perplexity degradation** (PPL: 6.06 -> 6.17).
*   **Stage 3 (Causal Patching):** *Success*.
*   **Stage 4 (Activation Healing):** *CATASTROPHIC FAILURE*. The "healing" algorithm destroyed the model, sending Perplexity from 6.17 up to **35,899.53** (+591,795%).

## Next Steps / Extensions
1.  **Diagnose Healing Failure:** The Stage 4 activation healing logic computes `pre_means` and `post_means` for all layers simultaneously, and then applies the difference as a hook to all layers simultaneously. This causes massive compounding distribution shifts because Layer N's shift completely invalidates the `post_means` calculated for Layer N+1.
2.  **Sequential Calibration:** We must rewrite Stage 4 to do Sequential Layer-by-Layer Calibration, similar to SparseGPT. We calibrate Layer 0, apply the hook, then use the *corrected* activations to calibrate Layer 1, apply the hook, and so on.
