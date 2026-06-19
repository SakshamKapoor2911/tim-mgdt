# Phase 14: Actionable Interpretability Pipeline
## Execution Run Notes

**Status:** `RUNNING` 🟢
**Initiated:** June 19, 2026, 05:04 AM
**Estimated Time to Completion:** ~1.5 - 2.0 hours (Across both models)

---

### Hardware & VRAM Profiling
The pipeline is running locally on the dual RTX 3090 setup (48GB VRAM total). 

**Utilization Snapshot (via `nvidia-smi`):**
- **GPU 0:** 244W / 350W (64% Compute Util) | VRAM: 8.2GB
- **GPU 1:** 257W / 350W (18% Compute Util) | VRAM: 8.3GB

> [!TIP]
> **VRAM Sharding Efficiency:** By using HuggingFace's `device_map="auto"`, we are perfectly balancing the 7B/8B models across both GPUs in `fp16` without triggering any out-of-memory errors. The 8.3GB per GPU footprint gives us massive headroom for the 500-step LoRA training buffers and gradient accumulation.

### Throughput Estimations
- **Oracle Sensitivity Measurement (Perplexity Knockout):** ~1.87 iterations per second. The full 32-layer Bathtub curve calculation will take approximately 3-4 minutes per model.
- **Surgical Recovery Training (500 steps):** Based on the Phase 13 timings (175s for 100 steps), each 500-step ablation will take ~15 minutes. 
- **Total Ablation Cost:** 3 ablations per model $\times$ 15m = ~45m per model.

### Defensive Architectural Controls
During setup, we prevented a critical bug: `nn.Identity` does not correctly handle the complex tuple unpacking (`hidden_states`, `attention_mask`, `past_key_value`) expected by HuggingFace Transformer layers. 

**Our Solution:** Instead of structural ablation, we perform **Weight Zeroing Ablation**. The `1_measure_sensitivity.py` script clones the output projection weights (`o_proj` and `down_proj`), zeroes them out in-place using `tensor.zero_()`, evaluates the $\Delta CE$, and restores them. This mathematically enforces the ablation without altering the internal graph architecture.
