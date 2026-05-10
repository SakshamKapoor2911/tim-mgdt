"""
Phase 10: Lottery Ticket Hypothesis Implementation
===================================================

4-Stage pipeline for identifying minimal winning subnetworks:
1. Ticket Identification - Find candidate winning tickets
2. Iterative Refinement - Binary search for sparsest winner
3. Pattern Analysis - Cross-model comparison
4. Deployment Metrics - Real-world efficiency measurement

Usage:
    from code.phases.phase10 import run_full_pipeline
    results = run_full_pipeline("gpt2", experiments_dir="experiments/phase10")
"""

from .stage1_identification import (
    prepare_wikitext_2,
    load_phase5_stability,
    generate_pruning_plans,
    create_pruned_model,
    validate_ticket,
    run_stage1
)

from .stage2_refinement import (
    binary_search_sparsity,
    run_stage2
)

from .stage3_pattern_analysis import (
    analyze_layer_importance_patterns,
    run_stage3
)

from .stage4_deployment_metrics import (
    measure_latency,
    measure_memory,
    compute_deployment_score,
    run_stage4
)

from .utils import (
    save_json,
    load_json,
    get_model_n_layers,
    rank_layers_by_importance
)

__all__ = [
    # Stage 1
    'prepare_wikitext_2',
    'load_phase5_stability',
    'generate_pruning_plans',
    'create_pruned_model',
    'validate_ticket',
    'run_stage1',
    # Stage 2
    'binary_search_sparsity',
    'run_stage2',
    # Stage 3
    'analyze_layer_importance_patterns',
    'run_stage3',
    # Stage 4
    'measure_latency',
    'measure_memory',
    'compute_deployment_score',
    'run_stage4',
    # Utils
    'save_json',
    'load_json',
    'get_model_n_layers',
    'rank_layers_by_importance',
]
