# The Perplexity Trap: Why Boundary Healing Fails Dense Reasoning, and the Case for Mechanistic Subnetwork Pruning

**Status**: Under Development (Pivot Branch)  
**Evaluated Models**: Meta-Llama-3-8B

---

## 🎯 Project Overview & Architectural Pivot

This branch documents a fundamental architectural pivot in our LLM compression thesis. 

Our earlier experiments successfully demonstrated that **Incision Boundary Healing** (targeting LoRA adapters exclusively to the layers adjacent to a dropped block) could rapidly recover the baseline validation loss and perplexity on generic pre-training distributions (e.g., C4).

However, rigorous zero-shot evaluation on downstream logic benchmarks (GSM8K, MMLU, ASDiv) yielded a stark **negative result**:

> ⚠️ **The Negative Result:** Recovering a validation loss curve does not equate to recovering a model's cognitive core. Macroscopic depth-pruning (dropping Layers 14-17) resulted in a $0\%$ recovery rate on structured reasoning tasks like GSM8K. Depth is a mandatory prerequisite for logic; width is redundant capacity.

We have officially pivoted the methodology to document this "Perplexity Trap" and introduce a new compression pipeline based on **Mechanistic Interpretability**.

### 🔬 The New Pipeline: Subnetwork Structural Excision

Instead of dropping entire sequential layers, we perform microscopic surgical excision across the network's width.

1.  **Empirical FIM Calibration:** We analyze the SwiGLU activation sparsity and Attention matrices using an Empirical Fisher Information Matrix (FIM) diagonal approximation over a pristine calibration dataset. This identifies the most globally salient reasoning circuits.
2.  **PyTorch Tensor Surgery:** We physically slice the Grouped-Query Attention (GQA) blocks (symmetrically pruning query heads to preserve topology) and the MLP intermediate dimensions, dynamically reshaping the `nn.Linear` matrices and bypassing standard HuggingFace configuration homogeneity.
3.  **Sparse LoRA Healing:** We identify the layers with the highest structural damage and inject highly targeted $r=16$ LoRA adapters, training exclusively on a blended reasoning manifold (GSM8K/MATH) with a strict zero-shot ASDiv quarantine.

---

## 📁 Codebase Organization

*   `paper/`: Contains the LaTeX source documenting the pivot to the Depth-Width Trade-off thesis.
*   `scratch/saliency_variant_analysis.py`: Evaluates first-order Taylor-expansion saliency mask stability (Spearman $\rho$).
*   `scratch/fim_variant_analysis.py`: The robust Empirical FIM calibration pipeline.
*   `scratch/prune_subnetworks.py`: The core PyTorch surgery script that executes asymmetric tensor slicing on GQA and MLP weights.
*   `scratch/diagnose_pruning_density.py`: Safetensor metadata parser to map the physical pruning distribution.
*   `scratch/sparse_lora_heal.py`: Custom dynamic loader and PEFT healing configuration.
*   `scratch/benchmark_pareto.py`: PyTorch native benchmarking script for MAC reduction and TPS (Tokens-Per-Second).
*   `archive/`: Contains legacy experiments on Layer-Dropping (Boundary Healing) which failed downstream reasoning evaluations.

---

## 🚀 Execution Pipeline

Our infrastructure is designed for multi-GPU execution (e.g., dual RTX 3090s) with strict memory management.

```bash
# 1. Establish the robust Mechanistic Saliency baseline (FIM)
python scratch/fim_variant_analysis.py

# 2. Slice the physical tensors based on structural sparsity
python scratch/prune_subnetworks.py

# 3. Diagnose the 6 most heavily damaged layers for PEFT targeting
python scratch/diagnose_pruning_density.py

# 4. Execute the targeted Sparse LoRA healing loop (ASDiv Quarantined)
python scratch/sparse_lora_heal.py

# 5. Extract hardware acceleration metrics (TPS & MACs)
python scratch/benchmark_pareto.py
```

> 🔍 **Negative Result Context:** The original Layer-Dropping methodology can still be explored in `code/phases/phase15_layer_drop/`, but its application is strictly limited to shallow semantic modeling, not dense logical reasoning.

---
*Created for COLM 2026 (Double-Blind Review)*
