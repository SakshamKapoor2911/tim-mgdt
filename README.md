# TIM-MGDT Implementation: Codebase Structure & Roadmap

**Project**: Temporal Importance Metrics for Model Gradient-based Diagnostic Tuning  
**Status**: Phase 6 Complete ✓ | Phases 7-9 Planned | Interventions Ready  
**Last Updated**: 2026-03-20

---

## 🎯 Project Overview

This repository implements a systematic framework for diagnosing and recovering from catastrophic layer sensitivity in transformer models, validated on GPT-2 and slated for Mistral-7B.

**Current Status:** [Phase Status Ledger](PHASE_STATUS_LEDGER.md)

---

## 📁 Codebase Organization

```
.
├── code/                          # All Python scripts organized by module
│   ├── phases/                    # Core logic by phase
│   ├── interventions/             # Reusable intervention modules
│   ├── metrics/                   # Diagnostic metrics
│   ├── models/                    # Model wrappers and registry
│   └── utils/                     # Shared utilities (dataset, manifest)
│
├── experiments/                   # Structured experiment results
│   ├── phase2/                    # Ground Truth Mapping
│   ├── phase5/                    # Advanced Proxies & Compression
│   ├── phase6/                    # Integration Validation
│   └── phase10/                   # Production Deployment Metrics
│
├── runners/                       # Standardized entrypoints for experiments
├── tests/                         # Unit and integration tests
├── docs/                          # Project documentation and phase summaries
│   ├── phases/                    # Summaries of individual phases
│   └── interventions/             # Summaries of intervention techniques
│
├── archive/                       # Historical docs and superseded code
├── venv/                          # Project virtual environment
├── requirements.txt
└── PHASE_STATUS_LEDGER.md         # Canonical source of phase completion
```

---

## 🚀 Quick Start

### 1. View Phase 6 Validation Results (Most Recent)

```bash
# Complete validation report
cat experiments/phase6/gpt2/wikitext2/baseline_legacy/phase6_validation_report.md

# Raw JSON results
cat experiments/phase6/gpt2/wikitext2/baseline_legacy/phase6_validation_results.json
```

### 2. Run Phase 6 Validation

```bash
python runners/large_model_validator.py
```

### 3. Explore Documentation

- **[Phase 6: Integration Validation](docs/phases/phase6_index.md)** ← START HERE
- **[Phase 5: Advanced Proxies](docs/phases/phase5_index.md)**
- **[Phase 2: Ground Truth](docs/phases/phase2_index.md)**

---

## 📊 Phase Roadmap

### ✅ Completed Phases

| Phase | Objective | Result Summary | Result File | Docs |
|-------|-----------|----------------|-------------|------|
| **Phase 5.1** | Hard pruning | ✗ Failed (expected) - PPL explodes (+108%) | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_1_benchmark_results.json` | [README](docs/phases/phase5_1.md) |
| **Phase 5.2** | Soft rebalancing | ✓ +0.31% PPL | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_2_conservative_recovery.json` | [README](docs/phases/phase5_2.md) |
| **Phase 5.3** | Progressive optimization | ✓ 0.98x optimal scaling | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_3_hybrid_strategy.json` | [README](docs/phases/phase5_3.md) |
| **Phase 6** | 500-sample validation | ✓ -0.11% PPL | `experiments/phase6/gpt2/wikitext2/baseline_legacy/phase6_validation_results.json` | [README](docs/phases/phase6.md) |

### 📋 Upcoming Phases

| Phase | Objective | Status | Timeline | Docs |
|-------|-----------|--------|----------|------|
| **Phase 7** | Alternative corpora | TODO | After 6 | [README](docs/phases/phase7_alternative_corpus.md) |
| **Phase 8** | Mistral-7B | BLOCKED (GPU) | Post-GPU | [README](docs/phases/phase8_mistral.md) |
| **Phase 9** | Downstream tasks | TODO | Post-7/8 | [README](docs/phases/phase9_downstream.md) |

---

## 🔧 Intervention Frameworks

**5 compression techniques** leveraging Phase 6 elasticity discovery:

| Intervention | Priority | Impact | ROI | Docs |
|--------------|----------|--------|-----|------|
| **Quantization Maps** | 🔴 HIGH | 75% size reduction | Huge | [README](docs/interventions/quantization_maps.md) ⭐ |
| **InAttention** | 🔴 HIGH | 4x speedup | Huge | [README](docs/interventions/inattention_conversion.md) ⭐ |
| **KL Divergence Proxy** | 🟡 MEDIUM | 100x profiling speedup | High | [README](docs/interventions/kl_divergence_proxy.md) |
| **Low-Rank Compression** | 🟡 MEDIUM | 20-40% size | Medium | [README](docs/interventions/low_rank_compression.md) |
| **Dynamic Early Exit** | 🟢 LOW | 1.5x speedup (subset) | Low | [README](docs/interventions/dynamic_early_exit.md) |

---

## 📚 Key Insights

### Phase 6 Validation (WikiText-2, 500 samples)

```
Baseline PPL:                51.43
With 0.98x Layer 10 scaling: 51.37
PPL Delta:                   -0.11%  ✓ IMPROVEMENT

Type: Reproduced from Phase 5.3 on 10x larger dataset
Confidence: 95% (n=321)
```

### Critical Discovery

> **Layer 10 is elastic, not broken.**

The +108% PPL explosion on hard removal shows over-weighting, not architectural damage. By scaling 2%, we restore balance.

---

## 📞 Getting Started

### To understand the project:
1. [Phase 6 README](docs/phases/phase6.md) (10 min)
2. [Phase 5.3 README](docs/phases/phase5_3.md) (10 min)
3. [Phase 5.1 README](docs/phases/phase5_1.md) - Why hard pruning fails (5 min)

### To implement an intervention:
1. Choose: [Quantization](docs/interventions/quantization_maps.md) (recommended)
2. Read detailed README (~20 min)
3. Follow implementation strategy
4. Use [KL Divergence Proxy](docs/interventions/kl_divergence_proxy.md) as foundation

### To run phases 7-9:
- **Phase 7**: [README](docs/phases/phase7_alternative_corpus.md)
- **Phase 8**: Wait for GPU APIs ([README](docs/phases/phase8_mistral.md))
- **Phase 9**: [README](docs/phases/phase9_downstream.md)

## 📊 Interpreting Results
After execution, results are saved in JSON format under `experiments/`.
*   **Spearman Rho (ρ):** We aim for > 0.7. This indicates our cheap proxies effectively rank layer sensitivity compared to the expensive PPL ground truth.
*   **P-Value:** Indicates statistical significance of the correlation (aim for < 0.05).

## 💡 Professional Tips for Execution
*   **Quantization:** The `HookedModelWrapper` uses `load_in_8bit=True` by default. If you have an A100 (40GB+), you can set this to `False` for higher precision.
*   **Checkpointing:** The current scripts save results at the end. For larger sweeps, consider adding intermediate JSON saves to prevent data loss on cloud preemptions.
*   **TransformerLens:** This project utilizes `HookedTransformer` for surgical activation extraction. Do not modify the `fold_ln` settings unless you are re-calibrating the Jacobians.

---

*Created by Algoverse (Team MGDT)*
