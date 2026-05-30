"""
Phase 11 Stage 1: Structured GQA-Aware Attention Head Pruning (REAL)

Computes per-head importance scores from actual Q-projection weight norms,
applies structured masks to zero out the least-important heads, and measures
the real perplexity impact on Wikitext-2 for each pruning plan.
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple, Any

from code.phases.phase11.utils import get_model_config, compute_perplexity


def _get_attn_projections(layer: nn.Module, arch: str):
    """Return (q_proj, k_proj, v_proj, o_proj) weight tensors for a layer."""
    if arch == "gpt2":
        # GPT-2 uses a fused c_attn projection: shape (3*hidden, hidden)
        # and c_proj for output
        return layer.attn.c_attn, None, None, layer.attn.c_proj
    else:
        # Llama / Mistral
        sa = layer.self_attn
        return sa.q_proj, sa.k_proj, sa.v_proj, sa.o_proj


def _score_heads_gpt2(model: nn.Module, cfg: Dict) -> Dict[str, float]:
    """
    Score attention heads for GPT-2.
    GPT-2 uses a fused c_attn weight of shape (3*hidden, hidden).
    The Q portion is rows [0 : hidden].
    """
    head_scores = {}
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    layers = cfg["layers"]

    for layer_idx, layer in enumerate(layers):
        # c_attn.weight shape: (hidden, 3*hidden) for Conv1D in GPT-2
        # We need to handle both Conv1D and Linear
        w = layer.attn.c_attn.weight.data
        # GPT-2 Conv1D stores weights as (in_features, out_features)
        # Q portion: columns [0 : hidden]
        if w.shape[0] < w.shape[1]:
            # Conv1D format: (in_features, 3*hidden)
            q_weight = w[:, :cfg["hidden_size"]]
        else:
            # Linear format: (3*hidden, in_features)
            q_weight = w[:cfg["hidden_size"], :]

        for head_idx in range(num_heads):
            start = head_idx * head_dim
            end = start + head_dim
            if q_weight.shape[0] == cfg["hidden_size"]:
                # Conv1D: slice columns
                score = torch.norm(q_weight[:, start:end], p=2).item()
            else:
                # Linear: slice rows
                score = torch.norm(q_weight[start:end, :], p=2).item()
            head_scores[f"L{layer_idx}_H{head_idx}"] = score

    return head_scores


def _score_heads_llama(model: nn.Module, cfg: Dict) -> Dict[str, float]:
    """
    Score attention heads for Llama/Mistral.
    Uses L2 norm of each head's Q-projection weight slice.
    """
    head_scores = {}
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    layers = cfg["layers"]

    for layer_idx, layer in enumerate(layers):
        q_proj = layer.self_attn.q_proj.weight.data
        for head_idx in range(num_heads):
            start = head_idx * head_dim
            end = start + head_dim
            score = torch.norm(q_proj[start:end, :], p=2).item()
            head_scores[f"L{layer_idx}_H{head_idx}"] = score

    return head_scores


def _apply_head_mask_llama(model: nn.Module, cfg: Dict, heads_to_prune: List[str]) -> None:
    """Zero out Q/K/V/O projection rows for pruned heads (Llama/Mistral)."""
    head_dim = cfg["head_dim"]
    layers = cfg["layers"]

    # Parse head keys into (layer_idx, head_idx)
    prune_set = set()
    for key in heads_to_prune:
        parts = key.split("_")
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        prune_set.add((layer_idx, head_idx))

    for layer_idx, layer in enumerate(layers):
        sa = layer.self_attn
        for head_idx in range(cfg["num_heads"]):
            if (layer_idx, head_idx) in prune_set:
                start = head_idx * head_dim
                end = start + head_dim
                sa.q_proj.weight.data[start:end, :] = 0.0
                sa.o_proj.weight.data[:, start:end] = 0.0
                # Zero corresponding KV heads (GQA-aware)
                kv_group_size = cfg["num_heads"] // cfg["num_kv_heads"]
                kv_head = head_idx // kv_group_size
                kv_start = kv_head * head_dim
                kv_end = kv_start + head_dim
                # Only zero KV if ALL query heads in this group are pruned
                all_in_group_pruned = all(
                    (layer_idx, kv_group_size * kv_head + q) in prune_set
                    for q in range(kv_group_size)
                )
                if all_in_group_pruned:
                    sa.k_proj.weight.data[kv_start:kv_end, :] = 0.0
                    sa.v_proj.weight.data[kv_start:kv_end, :] = 0.0


def _apply_head_mask_gpt2(model: nn.Module, cfg: Dict, heads_to_prune: List[str]) -> None:
    """Zero out attention head weights for GPT-2 (fused c_attn)."""
    head_dim = cfg["head_dim"]
    hidden = cfg["hidden_size"]
    layers = cfg["layers"]

    prune_set = set()
    for key in heads_to_prune:
        parts = key.split("_")
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        prune_set.add((layer_idx, head_idx))

    for layer_idx, layer in enumerate(layers):
        w = layer.attn.c_attn.weight.data
        for head_idx in range(cfg["num_heads"]):
            if (layer_idx, head_idx) in prune_set:
                start = head_idx * head_dim
                end = start + head_dim
                # GPT-2 Conv1D: (in, 3*hidden), Q is cols [0:hidden]
                if w.shape[0] < w.shape[1]:
                    w[:, start:end] = 0.0              # Q
                    w[:, hidden + start:hidden + end] = 0.0  # K
                    w[:, 2*hidden + start:2*hidden + end] = 0.0  # V
                else:
                    w[start:end, :] = 0.0
                    w[hidden + start:hidden + end, :] = 0.0
                    w[2*hidden + start:2*hidden + end, :] = 0.0
                # c_proj output rows
                c_proj_w = layer.attn.c_proj.weight.data
                if c_proj_w.shape[0] < c_proj_w.shape[1]:
                    c_proj_w[start:end, :] = 0.0
                else:
                    c_proj_w[:, start:end] = 0.0


def run_stage1(
    model: nn.Module,
    tokenizer: Any,
    device: str,
    model_name: str = "unknown",
    ppl_max_samples: int = 128,
) -> Dict[str, Any]:
    """
    Stage 1: Structured GQA-Aware Attention Head Pruning (REAL)

    1. Introspect model architecture
    2. Compute L2-norm importance score for every attention head
    3. For each sparsity plan (20%, 40%): zero out the lowest-scored heads,
       measure actual perplexity on Wikitext-2, then restore weights
    4. Return measured perplexity for each plan

    Raises ValueError if no model is loaded.
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Stage 1 requires a loaded model and tokenizer for real computation. "
            "Cannot proceed without a model."
        )

    print("\n" + "-" * 70)
    print("STAGE 1: Structured GQA-Aware Attention Head Pruning (REAL)")
    print("-" * 70)

    cfg = get_model_config(model)
    arch = cfg["arch_family"]
    print(f"  Architecture: {arch} | Layers: {cfg['num_layers']} | "
          f"Heads: {cfg['num_heads']} (KV: {cfg['num_kv_heads']}) | "
          f"Params: {cfg['num_params']/1e6:.1f}M")

    # --- Score heads ---
    print("  [1/3] Computing per-head L2 weight norm importance scores...")
    if arch == "gpt2":
        head_scores = _score_heads_gpt2(model, cfg)
    else:
        head_scores = _score_heads_llama(model, cfg)

    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1])
    total_heads = len(sorted_heads)
    print(f"    Scored {total_heads} heads. "
          f"Min: {sorted_heads[0][1]:.4f}, Max: {sorted_heads[-1][1]:.4f}")

    # --- Baseline perplexity ---
    print("  [2/3] Measuring baseline perplexity (no pruning)...")
    baseline_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)

    # --- Prune and measure ---
    print("  [3/3] Applying structured head masks and measuring perplexity...")
    prune_plans = {}

    # Save original state dict for restoration
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    for sparsity in [0.20, 0.40]:
        num_to_prune = int(total_heads * sparsity)
        heads_to_prune = [h[0] for h in sorted_heads[:num_to_prune]]

        # Apply mask
        if arch == "gpt2":
            _apply_head_mask_gpt2(model, cfg, heads_to_prune)
        else:
            _apply_head_mask_llama(model, cfg, heads_to_prune)

        # Measure perplexity with pruned heads
        pruned_ppl = compute_perplexity(model, tokenizer, device, max_samples=ppl_max_samples)
        ppl_increase_pct = ((pruned_ppl - baseline_ppl) / baseline_ppl) * 100

        prune_plans[f"gqa_prune_{int(sparsity*100)}"] = {
            "sparsity": sparsity,
            "heads_pruned": num_to_prune,
            "heads_kept": total_heads - num_to_prune,
            "gqa_kv_heads": cfg["num_kv_heads"],
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_increase_pct": ppl_increase_pct,
            "metric_type": "MEASURED",
        }

        print(f"  [OK] Plan gqa_prune_{int(sparsity*100)}: "
              f"PPL {baseline_ppl:.2f} -> {pruned_ppl:.2f} "
              f"(+{ppl_increase_pct:.2f}%)")

        # Restore original weights for next plan
        model.load_state_dict(original_state)

    return {
        "stage": "1_structured_pruning",
        "baseline_ppl": baseline_ppl,
        "plans_count": len(prune_plans),
        "plans": prune_plans,
        "head_score_range": {"min": sorted_heads[0][1], "max": sorted_heads[-1][1]},
        "metric_type": "MEASURED",
        "status": "complete",
    }
