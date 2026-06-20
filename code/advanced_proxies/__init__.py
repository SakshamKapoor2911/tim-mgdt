"""
Advanced Proxies Package.

Phase 5 components for granular compression:
  - Head-Level Jacobian (functional leverage per attention head)
  - Circuit Analyzer (lottery ticket identification)
  - Pruning Engine (mask-based model compression)
"""

from .head_jacobian import HeadJacobian, HeadJacobianConfig
from .circuit_analyzer import CircuitAnalyzer, TicketConfig
from .pruning_engine import PruningEngine, PruningConfig

__all__ = [
    "HeadJacobian",
    "HeadJacobianConfig",
    "CircuitAnalyzer",
    "TicketConfig",
    "PruningEngine",
    "PruningConfig",
]
