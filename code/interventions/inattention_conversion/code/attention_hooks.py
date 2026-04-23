"""
TransformerLens Attention Hooks for InAttention Conversion

Provides hook-based mechanism to replace standard attention with InAttention
using TransformerLens's hooking system.
"""

import torch
import torch.nn.functional as F
import math
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AttentionHookManager:
    """
    Manages hooks to replace attention with InAttention in TransformerLens.
    
    Strategy: Hook the attention output to force positions to attend only
    to position 0's key-value pair.
    """
    
    def __init__(self, model, layer_idx: int, device: str = "cuda"):
        """
        Initialize hook manager for specific layer.
        
        Args:
            model: HookedTransformer model
            layer_idx: Layer index to modify (0-11 for GPT-2)
            device: Device to use
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.hook_handles = []
        self.original_weights = {}
    
    def create_inattention_hook(
        self,
        scale_factor: float = 1.0,
        attend_pos: int = 0
    ) -> Callable:
        """
        Create a hook that forces attention to only attend to a specific position.
        
        Args:
            scale_factor: Scale factor to apply to modified attention
            attend_pos: Position to attend to (0 for first token)
        
        Returns:
            Hook function for use with TransformerLens
        """
        
        def inattention_hook(attn_pattern, hook):
            """
            Hook that modifies attention pattern to create InAttention.
            
            For hook_pattern, this is the post-softmax attention weights.
            Shape: [batch, n_heads, query_len, key_len]
            
            We force all query positions to attend only to a single key position.
            """
            if attn_pattern.dim() == 4:  # attn_pattern shape [batch, heads, q_len, k_len]
                batch_size, num_heads, q_len, k_len = attn_pattern.shape
                
                # Create new attention pattern: all queries attend only to position 0
                new_pattern = torch.zeros_like(attn_pattern)
                new_pattern[:, :, :, attend_pos] = 1.0
                
                return new_pattern * scale_factor
            
            return attn_pattern
        
        return inattention_hook
    
    def register_inattention_hook(
        self,
        hook_point: str = "pattern",
        scale_factor: float = 1.0,
        attend_pos: int = 0
    ) -> None:
        """
        Register the InAttention hook on the model.
        
        Uses TransformerLens hook system to modify attention patterns.
        
        Args:
            hook_point: 'pattern' for hook_pattern (post-softmax attention)
            scale_factor: Scale factor to apply
            attend_pos: Position to attend to (0 = first token)
        
        Returns:
            None (modifies model in-place)
        """
        hook_fn = self.create_inattention_hook(scale_factor, attend_pos)
        hook_name = f"blocks.{self.layer_idx}.attn.hook_{hook_point}"
        
        logger.info(f"Registering InAttention hook at: {hook_name}")
        try:
            # Add hook with permanent=False so it doesn't persist
            handle = self.model.add_hook(hook_name, hook_fn, is_permanent=False)
            self.hook_handles.append(handle)
            logger.info(f"  ✓ Hook registered successfully")
        except KeyError as e:
            logger.error(f"  ✗ Hook point not found. Available blocks should be 0-11. Error: {e}")
            self.hook_handles.append(None)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            if handle is not None:
                try:
                    handle.remove()
                except Exception as e:
                    logger.warning(f"Failed to remove hook: {e}")
        self.hook_handles.clear()
        logger.info("All InAttention hooks removed")
    
    def save_layer_weights(self) -> None:
        """Save original weights of the target layer."""
        for name, param in self.model.named_parameters():
            if f".{self.layer_idx}." in name:
                self.original_weights[name] = param.data.clone()
    
    def restore_layer_weights(self) -> None:
        """Restore original weights of the target layer."""
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data = self.original_weights[name].clone()
    
    def scale_attention_weights(self, scale_factor: float) -> None:
        """
        Scale all weights in attention layers.
        
        Args:
            scale_factor: Factor to scale by (0.98 = 2% reduction)
        """
        scaled_count = 0
        for name, param in self.model.named_parameters():
            if f"blocks.{self.layer_idx}.attn" in name and param.dim() > 0:
                param.data = param.data * scale_factor
                scaled_count += 1
        
        logger.info(f"Scaled {scaled_count} attention weight tensors by {scale_factor:.2f}x")


class HybridAttentionManager:
    """
    Manages multiple layers with different attention mechanisms.
    Can mix standard and InAttention layers for hybrid model.
    """
    
    def __init__(self, model):
        """
        Initialize manager for multiple layers.
        
        Args:
            model: HookedTransformer model
        """
        self.model = model
        self.layer_managers = {}
    
    def add_inattention_layer(
        self,
        layer_idx: int,
        scale_factor: float = 1.0,
        device: str = "cuda"
    ) -> AttentionHookManager:
        """
        Add InAttention to specific layer.
        
        Args:
            layer_idx: Layer to convert
            scale_factor: Scale factor for that layer
            device: Device to use
        
        Returns:
            AttentionHookManager for that layer
        """
        manager = AttentionHookManager(self.model, layer_idx, device)
        manager.register_inattention_hook(scale_factor=scale_factor)
        self.layer_managers[layer_idx] = manager
        return manager
    
    def cleanup(self) -> None:
        """Remove all hooks."""
        for manager in self.layer_managers.values():
            manager.remove_hooks()
        self.layer_managers.clear()


class MemoryProfiler:
    """Profile memory usage with and without InAttention."""
    
    @staticmethod
    def profile_layer_memory(
        model,
        layer_idx: int,
        input_shape: tuple,
        num_iterations: int = 5,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Profile memory usage of a single layer.
        
        Args:
            model: HookedTransformer model
            layer_idx: Layer to profile
            input_shape: Shape of input tensor (batch, seq_len, d_model)
            num_iterations: Number of forward passes to average
            device: Device to use
        
        Returns:
            Dictionary with memory statistics
        """
        import torch
        
        # Clear cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Create dummy input
        x = torch.randn(input_shape, device=device)
        
        # Record memory
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(x)
        
        # Actual profiling
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        
        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            peak_memory = 0
        
        return {
            "input_shape": input_shape,
            "seq_len": input_shape[1],
            "batch_size": input_shape[0],
            "peak_memory_bytes": peak_memory,
            "peak_memory_mb": peak_memory / (1024**2),
            "iterations": num_iterations
        }
    
    @staticmethod
    def compare_memory(
        model,
        layer_idx: int,
        input_shape: tuple,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Compare memory between standard and InAttention.
        
        Args:
            model: HookedTransformer model
            layer_idx: Layer to compare
            input_shape: Input shape
            device: Device to use
        
        Returns:
            Comparison dictionary
        """
        # Profile standard (no hooks)
        standard_profile = MemoryProfiler.profile_layer_memory(
            model, layer_idx, input_shape, device=device
        )
        
        # Profile with InAttention
        manager = AttentionHookManager(model, layer_idx, device)
        manager.register_inattention_hook()
        
        inattention_profile = MemoryProfiler.profile_layer_memory(
            model, layer_idx, input_shape, device=device
        )
        
        manager.remove_hooks()
        
        savings_mb = standard_profile["peak_memory_mb"] - inattention_profile["peak_memory_mb"]
        savings_pct = (savings_mb / standard_profile["peak_memory_mb"]) * 100 if standard_profile["peak_memory_mb"] > 0 else 0
        
        return {
            "standard_memory_mb": standard_profile["peak_memory_mb"],
            "inattention_memory_mb": inattention_profile["peak_memory_mb"],
            "savings_mb": savings_mb,
            "savings_percent": savings_pct,
            "input_shape": input_shape
        }


if __name__ == "__main__":
    print("AttentionHookManager test")
    
    # Example: would be used with actual model
    print("This module provides hooks for TransformerLens integration")
    print("See validate_single_layer.py for usage example")
