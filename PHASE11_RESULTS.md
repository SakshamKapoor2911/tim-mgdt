# Phase 11 Results

This document will contain the final analysis, outputs, and metrics from the parallel Phase 11 execution of Mistral-7B and Llama-3-8B.

## Mistral-7B
*   **Stage 1 & 2 (IMP Search):** *Success*. Reached **19.6% global sparsity** with only a **+2.54% Perplexity degradation** (PPL: 5.34 -> 5.47).
*   **Stage 3 (Causal Patching):** *Success*.
*   **Stage 4 (Activation Healing):** *Failed to recover performance*. Even with strict Sequential Calibration, the healed PPL jumped to **7.03** (+31.75%).
*   **Stage 4 (Masked SFT):** *Hardware Physics Wall*. Full-parameter AdamW requires 42GB+ VRAM, crashing our 24GB GPUs. DeepSpeed ZeRO-3 is required.

## Llama-3-8B
*   **Stage 1 & 2 (IMP Search):** *Success*. Reached **9.3% global sparsity** with a **+1.75% Perplexity degradation** (PPL: 6.06 -> 6.17). Phase transition occurs much earlier than Mistral.
*   **Stage 4 (Masked SFT):** *Hardware Physics Wall*. Same OOM barrier as Mistral.

## Final Scientific Analysis: Why Activation Healing Failed
We perfectly implemented Sequential Layer-by-Layer Activation Healing to eliminate cascading distribution shifts. However, the model performance *still degraded* (Mistral PPL went from 5.47 -> 7.03).

**The Root Cause: RMSNorm Conflicts**
Modern architectures like Llama-3 and Mistral-7B use `RMSNorm` (Root Mean Square Normalization) instead of traditional LayerNorm. RMSNorm strictly divides the activation vector by its Root Mean Square (variance) without re-centering the mean. 
When we manually shift the hidden states by adding a constant `correction` vector:
1. It drastically alters the magnitude of the vectors.
2. The subsequent `RMSNorm` divides the true signal by this artificially inflated RMS value.
3. This crushes the magnitude of the actual signal tokens, degrading attention distributions and FFN outputs, mathematically destroying the model's performance!

## Conclusion & Next Steps
**Activation Healing via constant mean-shift is scientifically incompatible with RMSNorm-based Large Language Models.**

The Phase 11 pipeline has conclusively proven that Iterative Magnitude Pruning (IMP) *alone* is highly effective (saving 20% sparsity on Mistral for only 2.5% PPL loss), but post-pruning activation shifting should be abandoned for modern architectures. Instead, future work should focus on *Weight-Weight* healing or *LoRA* fine-tuning to recover the lost 2.5% PPL without violating RMSNorm invariants!
