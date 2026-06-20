"""
InAttention Layer Implementation

Implements the InAttention mechanism that forces each token to attend only
to the initial token's keys/values, reducing attention from O(T²) to O(T).

Reference: InAttention - LINEAR CONTEXT SCALING FOR TRANSFORMERS
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any


class InAttentionHead(nn.Module):
    """
    Single attention head modified to InAttention mechanism.
    
    Instead of full softmax(QK^T)V, this computes:
    - Only dot products with position 0
    - Collapses attention to a scalar per head
    - Linear memory and computation
    """
    
    def __init__(self, d_head: int, device: str = "cuda"):
        super().__init__()
        self.d_head = d_head
        self.device = device
    
    def forward(
        self,
        q: torch.Tensor,           # [batch, seq_len, d_head]
        k: torch.Tensor,           # [batch, seq_len, d_head]
        v: torch.Tensor,           # [batch, seq_len, d_head]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InAttention: attend only to position 0.
        
        Args:
            q: Query tensor [batch, seq_len, d_head]
            k: Key tensor [batch, seq_len, d_head]
            v: Value tensor [batch, seq_len, d_head]
            mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, d_head] where all positions attend to position 0
        """
        batch_size, seq_len, _ = q.shape
        
        # Get keys and values for position 0 (only attend to this)
        k_0 = k[:, 0:1, :]           # [batch, 1, d_head]
        v_0 = v[:, 0:1, :]           # [batch, 1, d_head]
        
        # Compute attention scores with only position 0 key
        # Q @ K_0^T = [batch, seq_len, d_head] @ [batch, d_head, 1]
        #           = [batch, seq_len, 1]
        scores = torch.matmul(q, k_0.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.squeeze(-1)  # [batch, seq_len]
        
        # Attention weights (softmax not needed for single key, but for consistency)
        attn_weights = torch.ones_like(scores) / 1.0  # All equal weight to position 0
        
        # Apply to values
        # Result: each position gets the same value (v_0) weighted equally
        output = v_0.expand(batch_size, seq_len, self.d_head)
        
        return output


class InAttentionBlock(nn.Module):
    """
    Multi-head InAttention block that can replace standard attention.
    
    Reduces full self-attention (O(T²)) to linear attending-to-first (O(T)).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: str = "cuda"
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.device = device
        
        # Linear projections (same as standard attention)
        self.W_q = nn.Linear(d_model, d_model, device=device)
        self.W_k = nn.Linear(d_model, d_model, device=device)
        self.W_v = nn.Linear(d_model, d_model, device=device)
        self.W_o = nn.Linear(d_model, d_model, device=device)
        
        # InAttention heads
        self.heads = nn.ModuleList([
            InAttentionHead(self.d_head, device=device)
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        InAttention forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply InAttention heads
        head_outputs = []
        for i, head in enumerate(self.heads):
            h_out = head(
                Q[:, i],  # [batch, seq_len, d_head]
                K[:, i],  # [batch, seq_len, d_head]
                V[:, i]   # [batch, seq_len, d_head]
            )
            head_outputs.append(h_out)
        
        # Concatenate heads
        output = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, d_model]
        
        # Final projection
        output = self.W_o(output)
        
        return output


class InAttentionAnalyzer:
    """
    Analyze InAttention behavior and memory savings.
    """
    
    @staticmethod
    def estimate_memory_savings(
        seq_len: int,
        num_heads: int,
        d_head: int,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Estimate memory reduction from standard to InAttention.
        
        Standard attention KV-cache: O(batch * seq_len * num_heads * d_head * 2)
        InAttention KV-cache: O(batch * 1 * num_heads * d_head * 2)  # Only position 0
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            d_head: Dimension per head
            batch_size: Batch size
        
        Returns:
            Dictionary with memory estimates
        """
        # Standard attention KV-cache per sequence
        standard_kv_cache = batch_size * seq_len * num_heads * d_head * 2 * 4  # bytes (float32)
        
        # InAttention KV-cache (only position 0)
        inattention_kv_cache = batch_size * 1 * num_heads * d_head * 2 * 4
        
        savings = standard_kv_cache - inattention_kv_cache
        savings_pct = (savings / standard_kv_cache) * 100
        
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "d_head": d_head,
            "standard_kv_cache_bytes": standard_kv_cache,
            "standard_kv_cache_mb": standard_kv_cache / (1024**2),
            "inattention_kv_cache_bytes": inattention_kv_cache,
            "inattention_kv_cache_mb": inattention_kv_cache / (1024**2),
            "savings_bytes": savings,
            "savings_mb": savings / (1024**2),
            "savings_percent": savings_pct
        }


if __name__ == "__main__":
    # Example usage
    print("InAttention Layer Example\n")
    
    # GPT-2 config
    d_model = 768
    num_heads = 12
    seq_len = 1024
    batch_size = 1
    
    # Memory analysis
    analyzer = InAttentionAnalyzer()
    analysis = analyzer.estimate_memory_savings(seq_len, num_heads, d_model // num_heads, batch_size)
    
    print(f"Sequence Length: {analysis['seq_len']}")
    print(f"Batch Size: {analysis['batch_size']}")
    print(f"KV-Cache (Standard): {analysis['standard_kv_cache_mb']:.2f} MB")
    print(f"KV-Cache (InAttention): {analysis['inattention_kv_cache_mb']:.2f} MB")
    print(f"Memory Savings: {analysis['savings_mb']:.2f} MB ({analysis['savings_percent']:.1f}%)")
    print()
    
    # Simple forward pass test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device}")
    
    model = InAttentionBlock(d_model, num_heads, device=device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
