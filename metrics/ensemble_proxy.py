"""
Ensemble Proxy Metrics: Sign-corrected combinations of the proxy triad.

ED correlates positively with ground truth (higher ED = more important).
Jacobian and nAUDC correlate negatively (higher = less important when ablated).
All metrics are sign-corrected to a common polarity (higher = more important)
before combination.
"""


class EnsembleProxy:
    """Combines multiple proxies into sign-corrected ensemble metrics."""

    @staticmethod
    def _normalize_to_01(scores):
        """Normalize a dict of scores to [0, 1] range."""
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span < 1e-12:
            return {k: 0.5 for k in scores}
        return {k: (v - lo) / span for k, v in scores.items()}

    @staticmethod
    def _invert(scores):
        """Invert scores so that low becomes high (sign correction)."""
        vals = list(scores.values())
        hi = max(vals)
        return {k: hi - v for k, v in scores.items()}

    @staticmethod
    def compute_sign_corrected_mean(ed_scores, jacobian_scores, naudc_scores):
        """
        Three-way rank-aligned mean: ED + inverted-Jacobian + inverted-nAUDC.

        All metrics are normalized to [0,1] with a common polarity
        (higher = more important) then averaged.
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
        ED × inverted-Jacobian product.

        Both normalized to [0,1] with common polarity before multiplication.
        """
        ed_norm = EnsembleProxy._normalize_to_01(ed_scores)
        jac_inv = EnsembleProxy._normalize_to_01(EnsembleProxy._invert(jacobian_scores))

        return {layer: ed_norm[layer] * jac_inv[layer] for layer in ed_scores}

    @staticmethod
    def compute_weighted_sum(ed_scores, jacobian_scores, naudc_scores,
                              ed_weight=1/3, jac_weight=1/3, naudc_weight=1/3):
        """
        Weighted sum of sign-corrected, normalized metrics.

        Default: equal weights (1/3 each).
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
