# Phase 15: Oracle-Guided Layer Dropping & Activation Healing

To achieve true hardware compression (latency and memory reduction) instead of purely theoretical parameters, we implement **Phase 15: Oracle-Guided Layer Dropping**. 

By deleting a block of contiguous layers with the absolute lowest sensitivity scores (as determined by the causal $\Delta$CE bathtub curves), we physically shrink the model by **12.5%** (dropping 4 layers).

---

## 1. Quantitative Summary (Run 1: Global Sensitivity-Based Healing)

Our first experiment targets the **top 6 globally most sensitive layers** of the model for LoRA adaptation (Surgical LoRA) on Wikitext-2 for 500 samples (Rank=16, LR=1e-5).

| Metric | Mistral-7B-v0.1 | Meta-Llama-3-8B |
|--------|-----------------|-----------------|
| **Pruned Layer Window** | `[11, 12, 13, 14]` | `[8, 9, 10, 11]` |
| **Pruning Window Sensitivity** | `0.1314` (Deep Bath Valley) | `0.3021` (Moderate Valley) |
| **Inference Latency (Baseline)** | 33.14 ms | 32.61 ms |
| **Inference Latency (Dropped)** | 28.11 ms | 27.44 ms |
| **Physical Latency Reduction** | **-15.18%** | **-15.84%** |
| **Baseline C4 Perplexity** | 7.855 | 8.990 |
| **Dropped C4 Perplexity** | 9.930 | 14.953 |
| **Healed C4 Perplexity (Run 1)** | **9.730** | **13.697** |
| **PPL Recovered (Run 1)** | `+0.200` | `+1.257` |
| **Remaining PPL Gap (vs. Baseline)**| `+1.875` | `+4.706` |

---

## 2. Qualitative Evaluation

### What Passed:
1. **Physical Latency Gains**: A consistent **~15.5% reduction in latency** was achieved on both models. This directly correlates with proportional throughput scaling and KV-cache footprint reductions, proving that layer dropping is a practical path to actual compression.
2. **Oracle Sensitivity Alignment**: The perplexity degradation directly scales with the oracle's pruning window sensitivity score. Mistral's valley is deeper and less sensitive (`0.1314`), leading to only `9.93` post-drop perplexity. Llama-3's valley is more active (`0.3021`), leading to `14.95` post-drop perplexity. The oracle successfully isolates the safest regions to drop.

### What Failed / Under-performed:
1. **Incomplete Healing Recovery**: While Llama-3 saw some recovery (`-1.26` PPL), the remaining gap is still very wide (`+4.71` points). Mistral recovered almost nothing (`-0.20` PPL).
2. **The Global Target Flaw**:
   * Standard Surgical LoRA targets the globally most sensitive layers. For Mistral-7B, these are layers `[0, 1, 3, 29, 30, 31]`.
   * However, the layer drop occurs in the middle `[11, 12, 13, 14]`, joining layer 10 directly to layer 15. The representation mismatch is highly localized at the incision boundary.
   * Fine-tuning early input/output layers does not resolve the local discontinuity, leading to poor recovery.

---

## 3. Algorithmic Pivot: Boundary-Focused Surgical LoRA
To improve recovery, we will update the healing configurations to target the **incision boundary layers** directly surrounding the drop site:
* For Mistral (drop 11-14): Target original layers `[9, 10, 15, 16]`.
* For Llama-3 (drop 8-11): Target original layers `[6, 7, 12, 13]`.
We will run this comparison next and document the improved metrics.
