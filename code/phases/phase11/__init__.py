"""
Phase 11: Advanced Subnetwork & Compression Optimizations
===========================================================

Modular execution pipeline for advanced model compression:
1. Stage 1: Structured Pruning (Head & GQA)
2. Stage 2: IMP Search (Iterative Search & Calibration)
3. Stage 3: Causal Patching (MechInterp activation checks)
4. Stage 4: Activation Healing (Covariance-based healing)
"""

from .stage1_structured_pruning import run_stage1
from .stage2_imp_search import run_stage2
from .stage3_causal_patching import run_stage3
from .stage4_activation_healing import run_stage4

from .utils import (
    save_json,
    load_json,
    get_device
)

__all__ = [
    'run_stage1',
    'run_stage2',
    'run_stage3',
    'run_stage4',
    'save_json',
    'load_json',
    'get_device',
]
