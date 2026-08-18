"""Dose-response / planar-control statistical analysis.

For each dose angle, reads all exp_8_dose_angle_*_seed_*.json artifacts plus the
dense baseline per-prompt vector (exp_7_rotation_curve.json k0_dense), computes
per-seed exact two-tailed McNemar p-values against the dense baseline, and
combines the three independent seeds with Fisher's method and Stouffer's Z.

Pooling discordant counts across seeds (N=1500) is NOT performed: all seeds
evaluate the identical 500 questions, so pooled counts are pseudo-replicated.

Usage:
  python -m code.phases.phase25_causal_interventions.analyze_dose_controls \
      --output experiments/phase25_causal_interventions/dose_controls_analysis.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2, norm

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = ROOT / "experiments" / "phase25_causal_interventions"
ROTATION_CURVE = ARTIFACT_DIR / "exp_7_rotation_curve.json"
SEEDS = ["0", "1", "2"]


def mcnemar_exact(b, c):
    """Two-tailed exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, float(2.0 * binom.cdf(k, n, 0.5)))


def combine_fisher(ps):
    chi2_stat = -2.0 * sum(np.log(p) for p in ps)
    return chi2_stat, float(chi2.sf(chi2_stat, 2 * len(ps)))


def combine_stouffer(ps, directions):
    """Stouffer combination for two-tailed p-values.

    Each two-tailed p is converted to a signed one-sided z via
    z_i = Phi^{-1}(1 - p_i/2) * sign(direction_i), where direction is the sign
    of (c - b) discordance (dose-only minus dense-only), so the combined test
    is two-tailed about the signed mean z.
    """
    zs = [norm.ppf(1.0 - p / 2.0) * (1.0 if s >= 0 else -1.0)
          for p, s in zip(ps, directions)]
    z = np.mean(zs) * np.sqrt(len(zs))
    return float(z), float(2.0 * norm.sf(abs(z)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=str(ARTIFACT_DIR / "dose_controls_analysis.json"))
    args = parser.parse_args()

    dense = json.load(open(ROTATION_CURVE))["k0_dense"]["per_prompt"]
    assert len(dense) == 500, f"dense baseline must be N=500, got {len(dense)}"

    angles = set()
    for path in ARTIFACT_DIR.glob("exp_8_dose_angle_*_seed_*.json"):
        angles.add(json.load(open(path))["angle_deg"])

    results = {"dense_acc": round(100.0 * sum(dense) / len(dense), 2), "seeds": SEEDS}
    for angle in sorted(angles):
        per_seed = []
        accs = []
        for seed in SEEDS:
            path = ARTIFACT_DIR / f"exp_8_dose_angle_{angle}_seed_{seed}.json"
            if not path.exists():
                per_seed.append({"seed": seed, "status": "missing"})
                continue
            d = json.load(open(path))
            vec = d["vec"]
            assert len(vec) == 500, f"seed {seed}: expected N=500, got {len(vec)}"
            b = sum(1 for a, c in zip(dense, vec) if a and not c)   # dense-only
            c = sum(1 for a, c in zip(dense, vec) if c and not a)   # dose-only
            p = mcnemar_exact(b, c)
            per_seed.append({"seed": seed, "acc": d["acc"], "b_dense_only": b,
                             "c_dose_only": c, "n_discordant": b + c,
                             "p_exact": p})
            accs.append(d["acc"])
        ps = [s["p_exact"] for s in per_seed if "p_exact" in s]
        dirs = [s["c_dose_only"] - s["b_dense_only"] for s in per_seed
                if "p_exact" in s]
        entry = {"per_seed": per_seed}
        if len(ps) == 3:
            chi2_s, p_fisher = combine_fisher(ps)
            z_st, p_stouffer = combine_stouffer(ps, dirs)
            entry.update({
                "mean_acc": round(float(np.mean(accs)), 2),
                "half_range": round(float(np.ptp(accs) / 2.0), 2),
                "seed_accs": accs,
                "min_per_seed_p": round(min(ps), 4),
                "fisher": {"chi2": round(chi2_s, 4), "p": round(p_fisher, 4)},
                "stouffer": {"z": round(z_st, 4), "p": round(p_stouffer, 4)},
            })
        results[str(angle)] = entry
        print(f"angle {angle}: accs={accs} "
              f"per-seed p={[round(s['p_exact'], 4) for s in per_seed if 'p_exact' in s]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()