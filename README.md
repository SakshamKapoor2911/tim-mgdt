# TIM-MGDT Implementation: Codebase Structure & Roadmap

**Project**: Temporal Importance Metrics for Model Gradient-based Diagnostic Tuning  
**Status**: Phase 6 Complete ✓ | Phase 10 In Progress | Phases 7-9 Planned  
**Last Updated**: 2026-05-24

---

## 🎯 Project Overview

This repository implements a systematic framework for diagnosing and recovering from catastrophic layer sensitivity in transformer models, validated on GPT-2 and slated for Mistral-7B.

**Current Status:** [Phase Status Ledger](PHASE_STATUS_LEDGER.md)

---

## 📁 Codebase Organization

```
.
├── code/                          # All Python scripts organized by module
│   ├── phases/                    # Core logic by phase (2, 5, 6, 10, ...)
│   ├── interventions/             # Reusable intervention modules
│   ├── metrics/                   # Diagnostic metrics
│   ├── models/                    # Model wrappers and registry
│   └── utils/                     # Shared utilities (dataset, manifest)
│
├── experiments/                   # Structured experiment results
│   ├── phase2/                    # Ground Truth Mapping
│   ├── phase5/                    # Advanced Proxies & Compression
│   ├── phase6/                    # Integration Validation
│   └── phase10/                   # Lottery Ticket / deployment metrics
│
├── runners/                       # Standardized entrypoints for experiments
├── tests/                         # Unit, integration, and GPU test scripts
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

### 3. Run Phase 10 / Remote LLM Tests

```bash
# Primary entry point — toggle models in tests/run_remote_llm.py or via CLI
python tests/run_remote_llm.py --models mistral-7b,llama-3-8b --precision fp16

# Dual 3090: one model per GPU (parallel tmux sessions)
CUDA_VISIBLE_DEVICES=0 python tests/run_remote_llm.py --models mistral-7b
CUDA_VISIBLE_DEVICES=1 python tests/run_remote_llm.py --models llama-3-8b

# Synthetic multi-size GPU sweep (no HuggingFace download)
python tests/test_phase10_comprehensive.py
```

See [REMOTE_SETUP_GUIDE.md](REMOTE_SETUP_GUIDE.md) for full remote SSH workflow.

### 4. Explore Documentation

- **[Phase 6: Integration Validation](experiments/phase6/PHASE_INDEX.md)** ← START HERE
- **[Phase 5: Advanced Proxies](experiments/phase5/PHASE_INDEX.md)**
- **[Phase 2: Ground Truth](experiments/phase2/PHASE_INDEX.md)**

---

## 📊 Phase Roadmap

### ✅ Completed Phases

| Phase | Objective | Result Summary | Result File | Docs |
|-------|-----------|----------------|-------------|------|
| **Phase 5.1** | Hard pruning | ✗ Failed (expected) - PPL explodes (+108%) | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_1_benchmark_results.json` | [README](experiments/phase5_1/README.md) |
| **Phase 5.2** | Soft rebalancing | ✓ +0.31% PPL | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_2_conservative_recovery.json` | [README](experiments/phase5_2/README.md) |
| **Phase 5.3** | Progressive optimization | ✓ 0.98x optimal scaling | `experiments/phase5/gpt2/wikitext2/baseline_legacy/phase5_3_hybrid_strategy.json` | [README](experiments/phase5_3/README.md) |
| **Phase 6** | 500-sample validation | ✓ -0.11% PPL | `experiments/phase6/gpt2/wikitext2/baseline_legacy/phase6_validation_results.json` | [README](experiments/phase6/README.md) |
| **Phase 10** | Lottery ticket pipeline | ✓ GPU metrics recorded | `experiments/phase10/mistral7b_real_pytorch_results.json` | `code/phases/phase10/` |

### 📋 Upcoming Phases

| Phase | Objective | Status | Timeline | Docs |
|-------|-----------|--------|----------|------|
| **Phase 7** | Alternative corpora | TODO | After 6 | [README](experiments/phase7_alternative_corpus/README.md) |
| **Phase 8** | Mistral-7B | BLOCKED (GPU) | Post-GPU | [README](experiments/phase8_mistral/README.md) |
| **Phase 9** | Downstream tasks | TODO | Post-7/8 | [README](experiments/phase9_downstream/README.md) |

---

## 🔧 Intervention Frameworks

**5 compression techniques** leveraging Phase 6 elasticity discovery:

| Intervention | Priority | Impact | ROI |
|--------------|----------|--------|-----|
| **Quantization Maps** | 🔴 HIGH | 75% size reduction | Huge |
| **InAttention** | 🔴 HIGH | 4x speedup | Huge — [code](code/interventions/inattention_conversion/) |
| **KL Divergence Proxy** | 🟡 MEDIUM | 100x profiling speedup | High |
| **Low-Rank Compression** | 🟡 MEDIUM | 20-40% size | Medium |
| **Dynamic Early Exit** | 🟢 LOW | 1.5x speedup (subset) | Low |

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
1. [Phase 6 README](experiments/phase6/README.md) (10 min)
2. [Phase 5.3 README](experiments/phase5_3/README.md) (10 min)
3. [Phase 5.1 README](experiments/phase5_1/README.md) - Why hard pruning fails (5 min)

### To implement an intervention:
1. Start with InAttention: [code/interventions/inattention_conversion/](code/interventions/inattention_conversion/)
2. Follow the validation runners under `runners/`
3. Use KL divergence proxy metrics from `code/metrics/` as a profiling foundation

### To run phases 7-9:
- **Phase 7**: [README](experiments/phase7_alternative_corpus/README.md)
- **Phase 8**: Wait for GPU APIs ([README](experiments/phase8_mistral/README.md))
- **Phase 9**: [README](experiments/phase9_downstream/README.md)

## 📊 Interpreting Results
After execution, results are saved in JSON format under `experiments/`.
*   **Spearman Rho (ρ):** We aim for > 0.7. This indicates our cheap proxies effectively rank layer sensitivity compared to the expensive PPL ground truth.
*   **P-Value:** Indicates statistical significance of the correlation (aim for < 0.05).

## 💡 Professional Tips for Execution
*   **Quantization:** The `HookedModelWrapper` uses `load_in_8bit=True` by default. If you have an A100 (40GB+), you can set this to `False` for higher precision.
*   **Checkpointing:** Scripts save results at the end. For larger sweeps, consider adding intermediate JSON saves to prevent data loss on cloud preemptions.
*   **TransformerLens:** This project utilizes `HookedTransformer` for surgical activation extraction. Do not modify the `fold_ln` settings unless you are re-calibrating the Jacobians.

---

*Created by Algoverse (Team MGDT)*
