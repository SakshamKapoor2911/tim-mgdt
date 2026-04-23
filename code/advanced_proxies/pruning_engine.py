"""
Pruning Engine for Model Compression.

Implements layer and head masking based on sensitivity rankings to validate
whether proxy-identified unimportant components can actually be removed without
significant performance degradation.
"""

import torch
import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    mask_value: float = 0.0  # Value to use when masking (0 for zeroing, can be learnable)
    preserve_residual: bool = True  # Keep residual connections intact
    verbose: bool = True


class PruningEngine:
    """
    Surgical pruning tool: masks specific layers or heads without removing them.
    This allows benchmarking the impact of removing components.
    """

    def __init__(self, wrapper, config: Optional[PruningConfig] = None):
        self.wrapper = wrapper
        self.config = config or PruningConfig()
        self.active_masks: Dict[str, torch.Tensor] = {}
        self.hooks_added: List[str] = []

    def prune_bottom_k_layers(
        self, sensitivity_ranking: Dict[str, float], k: int = 3
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Mask the k least important layers (by sensitivity score).

        Args:
            sensitivity_ranking: Dictionary mapping "layer_X" to importance score
            k: Number of layers to prune (lowest scores)

        Returns:
            Tuple of (pruned_layer_indices, pruning_summary)
        """
        # Sort by sensitivity (ascending = least important first)
        sorted_layers = sorted(
            sensitivity_ranking.items(), key=lambda x: x[1]
        )
        
        to_prune = [int(layer_key.split("_")[1]) for layer_key, _ in sorted_layers[:k]]

        logger.info(f"🔪 Pruning bottom {k} layers by sensitivity: {to_prune}")

        pruning_summary = {}
        for layer_idx in to_prune:
            score = sensitivity_ranking.get(f"layer_{layer_idx}", 0.0)
            logger.info(f"  Layer {layer_idx}: score={score:.4f}")
            pruning_summary[f"layer_{layer_idx}"] = score

        # Apply masks
        self._apply_layer_masks(to_prune)

        return to_prune, pruning_summary

    def prune_bottom_percentage(
        self, sensitivity_ranking: Dict[str, float], percentage: float = 25.0
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Prune the bottom X% of layers by sensitivity.

        Args:
            sensitivity_ranking: Dictionary mapping "layer_X" to importance score
            percentage: Percentage of layers to prune (0-100)

        Returns:
            Tuple of (pruned_layer_indices, pruning_summary)
        """
        n_layers = max(int(layer_key.split("_")[1]) 
                      for layer_key in sensitivity_ranking.keys()) + 1
        k = max(1, int(n_layers * percentage / 100.0))

        logger.info(
            f"Pruning bottom {percentage}% of layers "
            f"({k} out of {n_layers} layers)"
        )

        return self.prune_bottom_k_layers(sensitivity_ranking, k=k)

    def _apply_layer_masks(self, layer_indices: List[int]):
        """
        Internal: Apply zero-masking to specified layers.
        """
        model = self.wrapper.model

        for layer_idx in layer_indices:
            try:
                # Hook into the residual post-attention output
                hook_name = f"blocks.{layer_idx}.attn.hook_z"

                def create_mask_hook(idx):
                    """Closure to capture layer index."""
                    def mask_hook(value, hook):
                        # Zero out attention output for this layer
                        return torch.zeros_like(value)
                    return mask_hook

                # Register hook
                model.add_hook(hook_name, create_mask_hook(layer_idx), is_permanent=True)
                self.hooks_added.append(hook_name)

                logger.debug(f"  Registered mask hook on {hook_name}")

            except Exception as e:
                logger.warning(f"Failed to mask layer {layer_idx}: {e}")

        if self.config.verbose:
            logger.info(f"Masks applied to {len(layer_indices)} layers")

    def prune_bottom_k_heads_per_layer(
        self,
        head_rankings: Dict[str, List[Tuple[int, float]]],
        k_per_layer: int = 3,
    ) -> Dict[str, List[int]]:
        """
        Mask the k least important attention heads within each layer.

        Args:
            head_rankings: Output from HeadJacobian.rank_heads_by_importance()
                          Maps "layer_X" -> [(head_idx, score), ...]
            k_per_layer: Number of heads to prune per layer

        Returns:
            Dictionary mapping "layer_X" to list of pruned head indices
        """
        logger.info(f"🔪 Pruning bottom {k_per_layer} heads per layer")

        pruned_heads = {}
        
        for layer_key, ranked_heads in head_rankings.items():
            layer_idx = int(layer_key.split("_")[1])
            
            # Get the k lowest-scoring heads (reverse sort, take from end)
            heads_to_prune = [head_idx for head_idx, score in ranked_heads[-k_per_layer:]]
            
            pruned_heads[layer_key] = heads_to_prune
            
            logger.info(
                f"  Layer {layer_idx}: Pruning heads {heads_to_prune} "
                f"(from {len(ranked_heads)} total)"
            )
            
            # Apply head masks
            self._apply_head_masks(layer_idx, heads_to_prune)

        return pruned_heads

    def _apply_head_masks(self, layer_idx: int, head_indices: List[int]):
        """
        Internal: Apply zero-masking to specific heads in a layer.
        """
        model = self.wrapper.model

        try:
            hook_name = f"blocks.{layer_idx}.attn.hook_z"

            def create_head_mask_hook(heads_to_mask):
                """Closure to capture heads to mask."""
                def mask_hook(value, hook):
                    # value shape: [batch, seq_len, n_heads, d_head]
                    masked_value = value.clone()
                    for head_idx in heads_to_mask:
                        masked_value[:, :, head_idx, :] = 0.0
                    return masked_value
                return mask_hook

            model.add_hook(hook_name, create_head_mask_hook(head_indices), is_permanent=True)
            self.hooks_added.append(hook_name)

            logger.debug(f"  Registered head mask hook on {hook_name}")

        except Exception as e:
            logger.warning(f"Failed to mask heads in layer {layer_idx}: {e}")

    def remove_all_masks(self):
        """
        Remove all pruning masks and restore original model.
        """
        model = self.wrapper.model
        
        for hook_name in self.hooks_added:
            try:
                model.remove_hook(hook_name)
                logger.debug(f"Removed hook: {hook_name}")
            except Exception as e:
                logger.warning(f"Failed to remove hook {hook_name}: {e}")

        self.hooks_added.clear()
        self.active_masks.clear()
        logger.info("All pruning masks removed. Model restored to baseline.")

    def get_pruning_statistics(self) -> Dict[str, int]:
        """
        Get statistics about current pruning state.

        Returns:
            Dictionary with pruning stats
        """
        stats = {
            "num_active_masks": len(self.hooks_added),
            "pruned_elements": len(self.active_masks),
            "masked_layers": list(set(
                int(h.split(".")[1]) for h in self.hooks_added
            )),
        }
        return stats

    def apply_custom_mask_function(
        self,
        layer_idx: int,
        mask_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Apply a custom masking function to a specific layer.

        Args:
            layer_idx: Layer index to mask
            mask_fn: Function that takes (value) and returns masked value
        """
        model = self.wrapper.model
        hook_name = f"blocks.{layer_idx}.attn.hook_z"

        def custom_mask_hook(value, hook):
            return mask_fn(value)

        model.add_hook(hook_name, custom_mask_hook, is_permanent=True)
        self.hooks_added.append(hook_name)
        logger.info(f"Applied custom mask function to layer {layer_idx}")
