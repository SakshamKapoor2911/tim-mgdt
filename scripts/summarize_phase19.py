"""Summarize Phase 19 mask-pruning controls for paper/dashboard use."""

import argparse
import glob
import json
import os
from pathlib import Path


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def gsm8k_metric(path):
    data = load_json(path)
    if not data:
        return None

    gsm8k = data.get("results", {}).get("gsm8k", {})
    samples = data.get("n-samples", {}).get("gsm8k", {})
    return {
        "path": path,
        "n": samples.get("effective", gsm8k.get("sample_len")),
        "strict": gsm8k.get("exact_match,strict-match"),
        "flex": gsm8k.get("exact_match,flexible-extract"),
    }


def best_gsm8k(pattern):
    metrics = [m for p in glob.glob(pattern) if (m := gsm8k_metric(p))]
    if not metrics:
        return {}
    return sorted(metrics, key=lambda m: (m.get("n") or 0, os.path.getmtime(m["path"])))[-1]


def pct(value):
    return "" if value is None else f"{value * 100:.2f}"


def number(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def collect_rows(exp_dir):
    mask_paths = sorted(
        glob.glob(os.path.join(exp_dir, "Mistral-7B-v0.1_mask_pruning_s*.json")),
        key=lambda p: (load_json(p) or {}).get("target_sparsity", 1),
    )
    dense_c4 = None
    if mask_paths:
        dense_c4 = (load_json(mask_paths[0]) or {}).get("baseline_c4_ppl")

    dense = best_gsm8k(os.path.join(exp_dir, "gsm8k_eval_baseline_full*.json"))
    dense_strict = dense.get("strict")
    rows = [
        {
            "tag": "dense",
            "target_sparsity_pct": 0.0,
            "actual_linear_sparsity_pct": 0.0,
            "c4_ppl": number(dense_c4),
            "gsm8k_n": dense.get("n"),
            "strict_em_pct": pct(dense.get("strict")),
            "flex_em_pct": pct(dense.get("flex")),
            "delta_strict_pt": "0.00" if dense_strict is not None else "",
            "metric_path": dense.get("path", ""),
        }
    ]

    for path in mask_paths:
        data = load_json(path) or {}
        tag = Path(path).stem.split("_")[-1]
        metric = best_gsm8k(os.path.join(exp_dir, f"gsm8k_eval_mask_{tag}*.json"))
        strict = metric.get("strict")
        delta = None if dense_strict is None or strict is None else (strict - dense_strict) * 100
        rows.append(
            {
                "tag": tag,
                "target_sparsity_pct": number((data.get("target_sparsity") or 0) * 100, 1),
                "actual_linear_sparsity_pct": number((data.get("linear_sparsity") or 0) * 100, 2),
                "c4_ppl": number(data.get("masked_c4_ppl")),
                "gsm8k_n": metric.get("n"),
                "strict_em_pct": pct(strict),
                "flex_em_pct": pct(metric.get("flex")),
                "delta_strict_pt": "" if delta is None else f"{delta:+.2f}",
                "metric_path": metric.get("path", ""),
            }
        )
    return rows


def print_markdown(rows):
    headers = [
        "tag",
        "target %",
        "actual linear %",
        "C4 PPL",
        "GSM8K n",
        "strict EM %",
        "flex EM %",
        "delta strict pt",
    ]
    keys = [
        "tag",
        "target_sparsity_pct",
        "actual_linear_sparsity_pct",
        "c4_ppl",
        "gsm8k_n",
        "strict_em_pct",
        "flex_em_pct",
        "delta_strict_pt",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/phase19")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
    args = parser.parse_args()

    rows = collect_rows(args.exp_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
