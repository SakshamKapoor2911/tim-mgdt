"""
Ensemble Proxy Metrics: Sign-corrected combinations of the proxy triad.

ED correlates positively with ground truth (higher ED = more important).
Jacobian and nAUDC correlate negatively (higher = less important when ablated).
All metrics are sign-corrected to a common polarity (higher = more important)
before combination.
"""


class EnsembleProxy:
    """Combines multiple proxies into sign-corrected ensemble metrics with numerical stability."""

    @staticmethod
    def _normalize_to_01(scores):
        """
        Normalize a dict of scores to [0, 1] range with numerical stability.
        
        Uses max(span, 1e-8) to avoid division by zero and prevent NaN propagation.
        Values outside [0, 1] are clipped to valid range.
        
        Args:
            scores: dict[str, float] with layer identifiers as keys
            
        Returns:
            dict[str, float]: Normalized scores in [0, 1]
        """
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-8)  # Prevent division by zero
        # Clip to [0, 1] to handle edge cases with outier-heavy distributions
        return {k: max(0.0, min(1.0, (v - lo) / span)) for k, v in scores.items()}

    @staticmethod
    def _invert(scores):
        """Invert scores so that low becomes high (sign correction).
        
        Computes: max(values) - value for each score, reversing the polarity.
        Used when metrics naturally increase with unimportance (e.g., Jacobian).
        
        Args:
            scores: dict[str, float]
            
        Returns:
            dict[str, float]: Sign-inverted scores
        """
        vals = list(scores.values())
        hi = max(vals)
        return {k: hi - v for k, v in scores.items()}

    @staticmethod
    def compute_sign_corrected_mean(ed_scores, jacobian_scores, naudc_scores):
        """
        Three-way rank-aligned mean: ED + inverted-Jacobian + inverted-nAUDC.

        All metrics are normalized to [0,1] with a common polarity
        (higher = more important) then averaged. Used for consensus importance ranking.
        
        Args:
            ed_scores: dict[str, float] - Effective dimension (positive correlation)
            jacobian_scores: dict[str, float] - Jacobian norm (inverted)
            naudc_scores: dict[str, float] - Propagation drift (inverted)
            
        Returns:
            dict[str, float]: Ensemble consensus scores
        """
        ed_norm = EnsembleProxy._normalize_to_01(ed_scores)
        jac_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(jacobian_scores))
        nau_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(naudc_scores))

        ensemble = {}
        for layer in ed_scores:
            ensemble[layer] = (ed_norm[layer] + jac_inv[layer] + nau_inv[layer]) / 3.0
        return ensemble

    @staticmethod
    def compute_sign_corrected_product(ed_scores, jacobian_scores):
        """
        ED × inverted-Jacobian product for multiplicative combination.

        Both normalized to [0,1] with common polarity before multiplication.
        Emphasizes agreement: only high if both ED is high AND Jacobian is low.
        
        Args:
            ed_scores: dict[str, float] - Effective dimension
            jacobian_scores: dict[str, float] - Jacobian norm (to be inverted)
            
        Returns:
            dict[str, float]: Multiplicative ensemble scores
        """
        ed_norm = EnsembleProxy._normalize_to_01(ed_scores)
        jac_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(jacobian_scores))

        return {layer: ed_norm[layer] * jac_inv[layer] for layer in ed_scores}

    @staticmethod
    def compute_weighted_sum(ed_scores, jacobian_scores, naudc_scores,
                              ed_weight=1/3, jac_weight=1/3, naudc_weight=1/3):
        """
        Weighted sum of sign-corrected, normalized metrics.

        Default: equal weights (1/3 each). Weights should sum to 1.0 for interpretability.
        
        Args:
            ed_scores: dict[str, float] - Effective dimension
            jacobian_scores: dict[str, float] - Jacobian norm (inverted)
            naudc_scores: dict[str, float] - Propagation drift (inverted)
            ed_weight: float - Weight for ED (default 1/3)
            jac_weight: float - Weight for Jacobian (default 1/3)
            naudc_weight: float - Weight for nAUDC (default 1/3)
            
        Returns:
            dict[str, float]: Weighted ensemble scores
        """
        ed_norm = EnsembleProxy._normalize_to_01(ed_scores)
        jac_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(jacobian_scores))
        nau_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(naudc_scores))

        ensemble = {}
        for layer in ed_scores:
            ensemble[layer] = (
                ed_weight * ed_norm[layer]
                + jac_weight * jac_inv[layer]
                + naudc_weight * nau_inv[layer]
            )
        return ensemble
