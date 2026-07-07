"""Bootstrap strict-match GSM8K confidence intervals for Phase 24 ablations."""

import argparse
import glob
import json
import os
import random
import re
import statistics
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


def include_sample_row(row, doc_id_limit=None):
    if doc_id_limit is None:
        return True
    doc_id = row.get("doc_id")
    return isinstance(doc_id, int) and doc_id < doc_id_limit


def strict_values(sample_path, doc_id_limit=None):
    values = []
    if not sample_path or not Path(sample_path).exists():
        return values
    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if not include_sample_row(row, doc_id_limit):
                continue
            if row.get("filter") != "strict-match":
                continue
            values.append(1 if row.get("exact_match") == 1.0 else 0)
    return values


def sample_exact_match(sample_path, filter_name, doc_id_limit=None):
    values = []
    if not sample_path or not Path(sample_path).exists():
        return None
    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if not include_sample_row(row, doc_id_limit):
                continue
            if row.get("filter") != filter_name:
                continue
            values.append(1 if row.get("exact_match") == 1.0 else 0)
    return mean(values)


def first_response(row):
    resps = row.get("resps") or []
    if not resps:
        return ""
    first = resps[0]
    if isinstance(first, list):
        return str(first[0]) if first else ""
    return str(first)


def mean(values):
    return None if not values else sum(values) / len(values)


def round_or_none(value, digits=3):
    return None if value is None else round(value, digits)


def generation_diagnostics(sample_path, doc_id_limit=None):
    diagnostics = {
        "sample_rows": 0,
        "strict_rows": 0,
        "avg_response_words": None,
        "median_response_words": None,
        "final_marker_rate": None,
        "extraction_failure_rate": None,
    }
    if not sample_path or not Path(sample_path).exists():
        return diagnostics

    word_counts = []
    marker_count = 0
    extraction_failure_count = 0

    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if not include_sample_row(row, doc_id_limit):
                continue
            diagnostics["sample_rows"] += 1
            if row.get("filter") != "strict-match":
                continue
            diagnostics["strict_rows"] += 1
            response = first_response(row)
            word_counts.append(len(re.findall(r"\S+", response)))
            if "####" in response:
                marker_count += 1
            filtered = row.get("filtered_resps") or []
            if not filtered or filtered[0] in ("", None):
                extraction_failure_count += 1

    strict_rows = diagnostics["strict_rows"]
    diagnostics.update(
        {
            "avg_response_words": round_or_none(mean(word_counts), 3),
            "median_response_words": round_or_none(statistics.median(word_counts), 3) if word_counts else None,
            "final_marker_rate": round_or_none(marker_count / strict_rows, 3) if strict_rows else None,
            "extraction_failure_rate": round_or_none(extraction_failure_count / strict_rows, 3)
            if strict_rows
            else None,
        }
    )
    return diagnostics


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
    row = {
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
    row.update(generation_diagnostics(sample_path))
    return row


def summarize_sample_baseline(tag, sample_path, result_path, meta_path, doc_id_limit, iters, seed):
    values = strict_values(sample_path, doc_id_limit=doc_id_limit)
    ci_low, ci_high = bootstrap_ci(values, iters=iters, seed=seed)
    meta = load_json(meta_path) or {}
    row = {
        "tag": tag,
        "result_path": result_path,
        "sample_path": sample_path,
        "source_doc_id_limit": doc_id_limit,
        "n": len(values),
        "strict": mean(values),
        "flex": sample_exact_match(sample_path, "flexible-extract", doc_id_limit=doc_id_limit),
        "sample_strict_n": len(values),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "actual_linear_sparsity": meta.get("linear_sparsity"),
        "masked_c4_ppl": meta.get("masked_c4_ppl"),
    }
    row.update(generation_diagnostics(sample_path, doc_id_limit=doc_id_limit))
    return row


def row_sort_key(row):
    order = {
        "s25_no_revert": 0,
        "revert_early": 1,
        "revert_mid": 2,
        "revert_late": 3,
        "revert_attention": 4,
        "revert_mlp": 5,
    }
    return order.get(row["tag"], 100), row["tag"]


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
        baseline_rows = [row for row in rows if row.get("tag") == "s25_no_revert" and row.get("strict") is not None]
        if baseline_rows:
            baseline = baseline_rows[0]
            baseline_gap_pt = (best_module["strict"] - baseline["strict"]) * 100
            return {
                "recommendation": "distributed_or_weak_module_signal",
                "best_module_group": best_module["tag"],
                "best_vs_second_gap_pt": round(gap_pt, 3),
                "best_vs_no_revert_gap_pt": round(baseline_gap_pt, 3),
                "detail": "Single-group dense reverts sit close to the same-prefix no-revert 25% baseline and have overlapping bootstrap intervals; report this as null or weak coarse-grained recovery, not a localization mechanism.",
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
        "| tag | n | strict % | 95% CI % | flex % | actual sparsity % | C4 PPL | avg words | final marker % | extract fail % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        ci = "n/a"
        if row.get("bootstrap_ci_low") is not None and row.get("bootstrap_ci_high") is not None:
            ci = f"{row['bootstrap_ci_low'] * 100:.1f}-{row['bootstrap_ci_high'] * 100:.1f}"
        lines.append(
            "| {tag} | {n} | {strict} | {ci} | {flex} | {sparsity} | {c4} | {avg_words} | {marker} | {extract_fail} |".format(
                tag=row["tag"],
                n=row.get("n") or "",
                strict="" if row.get("strict") is None else f"{row['strict'] * 100:.2f}",
                ci=ci,
                flex="" if row.get("flex") is None else f"{row['flex'] * 100:.2f}",
                sparsity="" if row.get("actual_linear_sparsity") is None else f"{row['actual_linear_sparsity'] * 100:.2f}",
                c4="" if row.get("masked_c4_ppl") is None else f"{row['masked_c4_ppl']:.3f}",
                avg_words="" if row.get("avg_response_words") is None else f"{row['avg_response_words']:.2f}",
                marker="" if row.get("final_marker_rate") is None else f"{row['final_marker_rate'] * 100:.1f}",
                extract_fail=""
                if row.get("extraction_failure_rate") is None
                else f"{row['extraction_failure_rate'] * 100:.1f}",
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
    parser.add_argument("--baseline_tag", default="")
    parser.add_argument("--baseline_sample", default="")
    parser.add_argument("--baseline_result", default="")
    parser.add_argument("--baseline_meta", default="")
    parser.add_argument("--baseline_doc_limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.result_glob))
    rows = [summarize_result(path, args.output_dir, args.bootstrap_iters, args.seed) for path in paths]
    if args.baseline_sample:
        rows.append(
            summarize_sample_baseline(
                args.baseline_tag or "baseline",
                args.baseline_sample,
                args.baseline_result,
                args.baseline_meta,
                args.baseline_doc_limit or None,
                args.bootstrap_iters,
                args.seed,
            )
        )
    rows = sorted(rows, key=row_sort_key)
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
