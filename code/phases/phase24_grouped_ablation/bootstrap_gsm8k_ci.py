"""Bootstrap strict-match GSM8K confidence intervals for Phase 24 ablations."""

import argparse
import glob
import json
import os
import random
from pathlib import Path


LAYER_ABLATIONS = ("revert_early", "revert_mid", "revert_late")
MODULE_ABLATIONS = ("revert_mlp", "revert_attention")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def gsm8k_metric(path):
    data = load_json(path) or {}
    gsm8k = data.get("results", {}).get("gsm8k", {})
    samples = data.get("n-samples", {}).get("gsm8k", {})
    return {
        "n": samples.get("effective", gsm8k.get("sample_len")) or 0,
        "strict": gsm8k.get("exact_match,strict-match"),
        "flex": gsm8k.get("exact_match,flexible-extract"),
    }


def sample_path_for_result(path):
    parent = Path(path).parent
    samples = sorted(parent.glob("samples_gsm8k_*.jsonl"), key=os.path.getmtime)
    return str(samples[-1]) if samples else None


def strict_values(sample_path):
    values = []
    if not sample_path or not Path(sample_path).exists():
        return values
    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("filter") != "strict-match":
                continue
            values.append(1 if row.get("exact_match") == 1.0 else 0)
    return values


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def bootstrap_ci(values, iters=2000, seed=42):
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        total = 0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def tag_for_result(path):
    parent_tag = Path(path).parent.name
    if parent_tag.startswith("revert_") or parent_tag == "s25":
        return parent_tag
    stem = Path(path).stem
    prefix = "gsm8k_eval_"
    if stem.startswith(prefix):
        return stem[len(prefix):].split("_limit", 1)[0]
    return stem


def metadata_for_result(output_dir, tag):
    paths = sorted(Path(output_dir).glob(f"*_grouped_mask_{tag}.json"), key=os.path.getmtime)
    return load_json(paths[-1]) if paths else {}


def summarize_result(path, output_dir, iters, seed):
    tag = tag_for_result(path)
    metric = gsm8k_metric(path)
    sample_path = sample_path_for_result(path)
    values = strict_values(sample_path)
    ci_low, ci_high = bootstrap_ci(values, iters=iters, seed=seed)
    meta = metadata_for_result(output_dir, tag) or {}
    return {
        "tag": tag,
        "result_path": path,
        "sample_path": sample_path,
        "n": metric["n"],
        "strict": metric["strict"],
        "flex": metric["flex"],
        "sample_strict_n": len(values),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "actual_linear_sparsity": meta.get("linear_sparsity"),
        "masked_c4_ppl": meta.get("masked_c4_ppl"),
    }


def decision(rows, min_delta_pt=4.0):
    layer_rows = [row for row in rows if row["tag"] in LAYER_ABLATIONS and row.get("strict") is not None]
    if len(layer_rows) < len(LAYER_ABLATIONS):
        return {
            "recommendation": "insufficient_layer_results",
            "detail": "Need early/mid/late layer ablations before deciding on contingency.",
        }

    module_rows = [row for row in rows if row["tag"] in MODULE_ABLATIONS and row.get("strict") is not None]
    if len(module_rows) == len(MODULE_ABLATIONS):
        ranked_modules = sorted(module_rows, key=lambda row: row["strict"], reverse=True)
        best_module = ranked_modules[0]
        second_module = ranked_modules[1]
        gap_pt = (best_module["strict"] - second_module["strict"]) * 100
        ci_nonoverlap = (
            best_module.get("bootstrap_ci_low") is not None
            and second_module.get("bootstrap_ci_high") is not None
            and best_module["bootstrap_ci_low"] > second_module["bootstrap_ci_high"]
        )
        if ci_nonoverlap or gap_pt >= min_delta_pt:
            return {
                "recommendation": "module_axis_signal",
                "best_module_group": best_module["tag"],
                "best_vs_second_gap_pt": round(gap_pt, 3),
                "detail": "Module-type dense revert has a clear enough leading group for a smoke-level mechanistic clue.",
            }
        return {
            "recommendation": "distributed_or_weak_module_signal",
            "best_module_group": best_module["tag"],
            "best_vs_second_gap_pt": round(gap_pt, 3),
            "detail": "Depth bands and module-type ablations have overlapping bootstrap intervals; report this as distributed or weakly localized recovery, not a clean mechanism.",
        }

    ranked = sorted(layer_rows, key=lambda row: row["strict"], reverse=True)
    best = ranked[0]
    second = ranked[1]
    gap_pt = (best["strict"] - second["strict"]) * 100
    ci_nonoverlap = (
        best.get("bootstrap_ci_low") is not None
        and second.get("bootstrap_ci_high") is not None
        and best["bootstrap_ci_low"] > second["bootstrap_ci_high"]
    )
    if ci_nonoverlap or gap_pt >= min_delta_pt:
        return {
            "recommendation": "stop_after_layer_groups",
            "best_layer_group": best["tag"],
            "best_vs_second_gap_pt": round(gap_pt, 3),
            "detail": "Layer-block ablation has a clear enough leading group for a smoke-level mechanistic clue.",
        }
    return {
        "recommendation": "run_module_contingency",
        "best_layer_group": best["tag"],
        "best_vs_second_gap_pt": round(gap_pt, 3),
        "detail": "Layer-block ablations are too close; run MLP-vs-attention dense-revert contingency.",
    }


def write_markdown(summary, path):
    lines = [
        "# Phase 24 Grouped Ablation Summary",
        "",
        "| tag | n | strict % | 95% CI % | flex % | actual sparsity % | C4 PPL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        ci = "n/a"
        if row.get("bootstrap_ci_low") is not None and row.get("bootstrap_ci_high") is not None:
            ci = f"{row['bootstrap_ci_low'] * 100:.1f}-{row['bootstrap_ci_high'] * 100:.1f}"
        lines.append(
            "| {tag} | {n} | {strict} | {ci} | {flex} | {sparsity} | {c4} |".format(
                tag=row["tag"],
                n=row.get("n") or "",
                strict="" if row.get("strict") is None else f"{row['strict'] * 100:.2f}",
                ci=ci,
                flex="" if row.get("flex") is None else f"{row['flex'] * 100:.2f}",
                sparsity="" if row.get("actual_linear_sparsity") is None else f"{row['actual_linear_sparsity'] * 100:.2f}",
                c4="" if row.get("masked_c4_ppl") is None else f"{row['masked_c4_ppl']:.3f}",
            )
        )
    lines.extend(["", f"Decision: `{summary['decision']['recommendation']}`", summary["decision"]["detail"], ""])
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_glob", required=True)
    parser.add_argument("--output_dir", default="experiments/phase24_grouped_ablation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default="")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.result_glob))
    rows = [summarize_result(path, args.output_dir, args.bootstrap_iters, args.seed) for path in paths]
    rows = sorted(rows, key=lambda row: row["tag"])
    summary = {
        "schema": "phase24_grouped_ablation_summary_v1",
        "result_glob": args.result_glob,
        "rows": rows,
        "decision": decision(rows),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    if args.markdown:
        write_markdown(summary, args.markdown)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
