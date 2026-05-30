"""
Phase 11: Real Pruning & Measurement Pipeline
==============================================

All stages perform real model computation with measured metrics.
No synthetic fallbacks, no hardcoded constants.

Pipeline stages:
  1. Stage 1: Structured GQA-Aware Head Pruning (measured PPL)
  2. Stage 2: Iterative Magnitude Pruning (measured PPL at each sparsity)
  3. Stage 3: Causal Activation Patching (real forward-hook interventions)
  4. Stage 4: Activation Healing (bias correction + measured deployment metrics)
"""

from .stage1_structured_pruning import run_stage1
from .stage2_imp_search import run_stage2
from .stage3_causal_patching import run_stage3
from .stage4_activation_healing import run_stage4

from .utils import (
    save_json,
    load_json,
    get_device,
    get_model_config,
    compute_perplexity,
    apply_global_pruning,
    remove_global_pruning,
    count_nonzero_params,
    time_forward_pass,
)

__all__ = [
    'run_stage1',
    'run_stage2',
    'run_stage3',
    'run_stage4',
    'save_json',
    'load_json',
    'get_device',
    'get_model_config',
    'compute_perplexity',
    'apply_global_pruning',
    'remove_global_pruning',
    'count_nonzero_params',
    'time_forward_pass',
]
