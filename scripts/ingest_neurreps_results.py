"""
Ingest NeurReps N=500 (8-shot) results into a single mapping for main.tex updates.

Reads the fresh result JSONs produced by run_neurreps_full_pipeline.sh and prints a
flat key -> value map (with CIs) so the paper update uses ONLY the new numbers.
Run locally on the pulled repo, or on the server after the pipeline finishes.
"""

import json
import os

BASE = "experiments/phase25_causal_interventions"

def load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(v):
    return f"{v:.1f}%"


def main():
    # --- P1: sufficiency + random control (N=500, 8-shot) ---
    suf = load("exp_1b_sufficiency.json")
    if suf:
        print("== P1 SUFFICIENCY (exp_1b_sufficiency.json) ==")
        print(f"  dense_baseline            = {fmt_pct(suf.get('dense_baseline', 0))}")
        print(f"  random_control (mean 3)   = {fmt_pct(suf.get('acc_random_rotated_control', 0))}")
        print(f"  rotated (R*)^T            = {fmt_pct(suf.get('acc_rotated_sufficiency', 0))}")
        print(f"  n = {suf.get('num_eval_samples')}")

    pr = load("exp_1a_procrustes.json")
    if pr:
        print("== P1 PROCRUSTES (exp_1a_procrustes.json) ==")
        print(f"  pruned_baseline           = {fmt_pct(pr.get('acc_pruned_baseline', 0))}")
        print(f"  procrustes_intervened     = {fmt_pct(pr.get('acc_procrustes_intervened', 0))}")

    mc = load("exp_mcnemar.json")
    if mc:
        print("== P1 MCNEMAR (exp_mcnemar.json) ==")
        for k, v in mc.items():
            if isinstance(v, dict) and "p_value" in v:
                print(f"  {k}: p={v['p_value']} (b={v.get('b')}, c={v.get('c')})")

    # --- P6: PC ablation ---
    pc = load("exp_6_pc_ablation.json")
    if pc:
        print("== P6 PC ABLATION (exp_6_pc_ablation.json) ==")
        for k in ["intact_baseline", "ablate_pc1", "ablate_pc2", "ablate_pc8",
                  "ablate_pc3_pc5", "ablate_pc4_6_7_9_10"]:
            v = pc.get(k)
            if v:
                print(f"  {k:24s} = {v['accuracy']:.1f}% [{v['ci_95'][0]:.1f}, {v['ci_95'][1]:.1f}] ({v['correct']}/{v['total']})")

    # --- P7: rotation curve ---
    rc = load("exp_7_rotation_curve.json")
    if rc:
        print("== P7 ROTATION CURVE (exp_7_rotation_curve.json) ==")
        for k in ["k0_dense", "k1", "k2", "k3", "k4"]:
            v = rc.get(k)
            if v:
                print(f"  {k:10s} theta={v.get('theta_deg', 0):.2f} deg  gsm8k={v['gsm8k']:.1f}% [{v['ci_95'][0]:.1f}, {v['ci_95'][1]:.1f}]")

    # --- P4: location sweep ---
    loc = load("exp_4_location_sweep.json")
    if loc:
        print("== P4 LOCATION SWEEP (exp_4_location_sweep.json) ==")
        for lk, lv in (loc.get("locations") or {}).items():
            print(f"  {lk:10s} = {lv['mean']:.2f}% [{lv['clopper_pearson_ci_95'][0]:.2f}, {lv['clopper_pearson_ci_95'][1]:.2f}] ({lv['total_correct']}/{lv['total_evaluated']})")

    # --- P5: PC1 un-rotation ---
    p5 = load("exp_5_pc1_unrotation.json")
    if p5:
        print("== P5 PC1 (exp_5_pc1_unrotation.json) ==")
        print(f"  theta_acute   = {p5.get('pc1_theta_acute')}")
        print(f"  theta_obtuse  = {p5.get('pc1_theta_obtuse')}")
        print(f"  proj_synced   = {p5.get('pc1_projection_synced')}")
        print(f"  proj_raw      = {p5.get('pc1_alignment_projection')}")
        print(f"  dense_base    = {p5.get('baseline_dense')}")
        print(f"  pruned_base   = {p5.get('baseline_pruned')}")
        print(f"  pc1_realigned = {p5.get('pc1_unrotation_gsm8k')}")

    # --- Energy collapse ---
    en = load("energy_collapse.json")
    if en:
        print("== ENERGY COLLAPSE (energy_collapse.json) ==")
        for model, v in en.items():
            pp = v.get("per_passage", {})
            cc = v.get("concatenated", {})
            print(f"  {model:30s} per-passage collapse = {pp.get('mean_energy_collapse_percent')}% +/- {pp.get('ci_energy_collapse_percent')}  "
                  f"theta = {pp.get('mean_theta_deg')} +/- {pp.get('ci_theta_deg')}  | concat = {cc.get('energy_collapse_percent')}%")

    print("\n[DONE] ingest complete")


if __name__ == "__main__":
    main()
