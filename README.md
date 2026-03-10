# TIM-MGDT Implementation: Diagnostic Proxies for Accelerated Compression

This directory contains the core implementation of the **Predictive Ranking Framework** designed to estimate LLM layer sensitivity using mechanistic diagnostic proxies.

## 📂 Project Structure

```text
implementation/
├── models/             # Model loading and TransformerLens wrappers
│   └── model_wrapper.py    # HookedTransformer wrapper for Mistral-7B
├── metrics/            # The "Proxy Triad" logic
│   ├── geometric_ed.py     # Randomized Effective Dimension (SVD logic)
│   ├── causal_jacobian.py  # Logit Jacobian Norm calculation
│   └── propagation_naudc.py # cumulative cosine drift (nAUDC)
├── interventions/      # Experimental modifications
│   └── ablation.py         # Ground Truth Mapping (Activation Ablation)
├── experiments/        # Raw data and result logging
│   └── sensitivity_results.json # Final correlation scores
├── visualizations/     # Plotting and analysis scripts (TBD)
├── setup.sh            # Automation script for environment setup
├── requirements.txt    # Stable dependencies for research environment
└── run_benchmarks.py   # Master Execution Engine
```

## 🛠️ File/Folder Functions

- **`models/model_wrapper.py`**: Handles the loading of Mistral-7B-v0.1 into `TransformerLens`. It enables 8-bit quantization to ensure the model fits on commodity GPUs.
- **`metrics/`**: This is the "brain" of the project. It houses the three orthogonal metrics used to predict sensitivity without exhaustive benchmarks.
- **`interventions/ablation.py`**: Used to establish our **Medium-Fidelity Ground Truth**. It performs zero-ablation on each layer to measure the resulting Perplexity (PPL) shift.
- **`run_benchmarks.py`**: The control tower. It orchestrates the model loading, runs the ground truth ablation sweep, calculates the proxy metrics, and computes the Spearman Correlation (ρ) between them.

## 🚀 Getting Started

### 1. Prerequisites (Hardware)
To run this perfectly, we recommend the following:
*   **Minimum GPU:** 16GB VRAM (e.g., NVIDIA T4, A10G, or RTX 4080).
*   **Recommended GPU:** 24GB+ VRAM (e.g., NVIDIA A10, RTX 3090/4090, or A100) for faster SVD calculations and larger batch sizes.
*   **Storage:** 20GB free space for model weights.

### 2. Environment Setup
Run the following commands to initialize your environment:
```bash
cd implementation
bash setup.sh
```

### 3. Running the Pipeline
Execute the master script to run the experimental matrix:
```bash
python run_benchmarks.py
```

## 📊 Interpreting Results
After execution, results are saved in `experiments/sensitivity_results.json`. 
*   **Spearman Rho (ρ):** We aim for > 0.7. This indicates our cheap proxies effectively rank layer sensitivity compared to the expensive PPL ground truth.
*   **P-Value:** Indicates statistical significance of the correlation (aim for < 0.05).

## 💡 Professional Tips for Execution
*   **Quantization:** The `HookedModelWrapper` uses `load_in_8bit=True` by default. If you have an A100 (40GB+), you can set this to `False` for higher precision.
*   **Checkpointing:** The current `run_benchmarks.py` saves results at the end. For larger sweeps, consider adding intermediate JSON saves to prevent data loss on cloud preemptions.
*   **TransformerLens:** This project utilizes `HookedTransformer` for surgical activation extraction. Do not modify the `fold_ln` settings unless you are re-calibrating the Jacobians.

---

*Created by Algoverse (Team MGDT)*