"""
Circuit Analyzer for Identifying "Winning Tickets" (Lottery Ticket Hypothesis).

Combines Jacobian (importance), nAUDC (stability), and ED (focus) to identify
which sub-networks are critical and which are redundant.

Formula: Stability = (Jacobian / nAUDC) × (1 / ED)
"""

import torch
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TicketConfig:
    """Configuration for lottery ticket scoring."""
    min_naudc: float = 1e-9  # Prevent division by zero
    stability_threshold: Optional[float] = None  # For filtering candidates
    verbose: bool = True


class CircuitAnalyzer:
    """
    Analyzes circuits using the Proxy Triad to identify spending lottery tickets.
    """

    def __init__(self, config: Optional[TicketConfig] = None):
        self.config = config or TicketConfig()

    def find_winning_tickets(
        self,
        ed_dict: Dict[str, float],
        jacobian_dict: Dict[str, float],
        naudc_dict: Dict[str, float],
        n_layers: int,
    ) -> Dict[str, float]:
        """
        Compute Stability Scores to identify "Winning Tickets":
        Stability = (Jacobian / nAUDC) × (1 / ED)

        High Stability = Layer/Head is:
          • Important (high Jacobian)
          • Reliable (low nAUDC - stable under perturbation)
          • Focused (low ED - not smeared across whole space)

        Args:
            ed_dict: Effective Dimension scores per layer
            jacobian_dict: Causal Jacobian scores per layer
            naudc_dict: Propagation nAUDC scores per layer
            n_layers: Total number of layers

        Returns:
            Dictionary mapping "layer_X" to stability score
        """
        stability_scores = {}

        logger.info("Computing Stability Scores (Lottery Ticket Identification)...")

        for i in range(n_layers):
            key = f"layer_{i}"
            j = jacobian_dict.get(key, 0.0)
            e = ed_dict.get(key, 1.0)
            n = max(naudc_dict.get(key, 0.0), self.config.min_naudc)

            # Avoid division by zero for E.D.
            e_safe = max(e, 0.1)

            # Stability formula
            stability = (j / n) * (1.0 / e_safe)
            stability_scores[key] = float(stability)

            if self.config.verbose:
                logger.debug(
                    f"Layer {i}: J={j:.2e}, nAUDC={n:.2e}, ED={e:.2e} → Stability={stability:.2e}"
                )

        logger.info(f"Stability Scores computed for {len(stability_scores)} layers")
        return stability_scores

    @staticmethod
    def rank_by_stability(
        stability_scores: Dict[str, float], ascending: bool = False
    ) -> list:
        """
        Rank layers by stability (for pruning decisions).

        Args:
            stability_scores: Stability scores per layer
            ascending: If True, sort low-to-high (prune top losers first)

        Returns:
            List of (layer_name, score) tuples, sorted by score
        """
        ranked = sorted(
            stability_scores.items(), key=lambda x: x[1], reverse=ascending
        )
        return ranked

    @staticmethod
    def identify_lottery_ticket_candidates(
        stability_scores: Dict[str, float],
        jacobian_dict: Dict[str, float],
        ed_dict: Dict[str, float],
        percentile_threshold: float = 75.0,
    ) -> Dict[str, Dict]:
        """
        Identify layers that are "Lottery Ticket" candidates.
        Criteria:
          • High Jacobian (top percentile)
          • Low ED (bottom percentile) → concentrated circuit

        Args:
            stability_scores: Stability scores
            jacobian_dict: Jacobian scores
            ed_dict: ED scores
            percentile_threshold: Percentile for thresholding

        Returns:
            Dictionary mapping layer_X to metadata about ticket status
        """
        candidates = {}

        # Compute percentiles
        jacobian_vals = list(jacobian_dict.values())
        ed_vals = list(ed_dict.values())

        j_threshold = torch.tensor(jacobian_vals).quantile(percentile_threshold / 100.0).item()
        ed_threshold = torch.tensor(ed_vals).quantile((100 - percentile_threshold) / 100.0).item()

        logger.info(
            f"Lottery Ticket Thresholds: "
            f"Jacobian > {j_threshold:.2e}, ED < {ed_threshold:.2f}"
        )

        for layer_key in stability_scores.keys():
            j = jacobian_dict.get(layer_key, 0.0)
            e = ed_dict.get(layer_key, 0.0)
            s = stability_scores[layer_key]

            is_high_j = j > j_threshold
            is_low_ed = e < ed_threshold
            is_ticket = is_high_j and is_low_ed

            candidates[layer_key] = {
                "stability": s,
                "jacobian": j,
                "ed": e,
                "is_high_jacobian": is_high_j,
                "is_low_ed": is_low_ed,
                "is_lottery_ticket": is_ticket,
            }

            if is_ticket:
                logger.info(f"  ✓ TICKET: {layer_key} (J={j:.2e}, ED={e:.2f}, S={s:.2e})")
            elif is_high_j:
                logger.debug(f"  ~ IMPORTANT but diffuse: {layer_key} (ED too high)")
            elif is_low_ed:
                logger.debug(f"  ~ FOCUSED but unimportant: {layer_key} (J too low)")

        return candidates

    @staticmethod
    def compute_pruning_priority(
        stability_scores: Dict[str, float], reverse: bool = False
    ) -> list:
        """
        Compute layer pruning priority (lowest stability → prune first).

        Args:
            stability_scores: Stability scores
            reverse: If True, highest stability first (keep these)

        Returns:
            Sorted list of layer indices for pruning
        """
        ranked = sorted(
            stability_scores.items(), key=lambda x: x[1], reverse=reverse
        )
        layer_indices = [int(k.split("_")[1]) for k, v in ranked]
        return layer_indices
