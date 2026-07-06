"""Check whether Phase 19 has enough evidence for a workshop-ready draft."""

import argparse
import glob
import json
import os
from pathlib import Path


REQUIRED_ARTIFACTS = [
    "paper/phase19_efficient_reasoning_draft.tex",
    "paper/phase19_workshop_outline.txt",
    "paper/figures/phase19_reasoning_dissociation.png",
    "dashboard/index.html",
    "dashboard/figures/phase19_reasoning_dissociation.png",
    "experiments/phase19/gsm8k_sample_analysis.json",
]


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
    strict = gsm8k.get("exact_match,strict-match")
    flex = gsm8k.get("exact_match,flexible-extract")
    if strict is None:
        return None
    return {
        "path": path,
        "n": samples.get("effective", gsm8k.get("sample_len")) or 0,
        "strict": strict,
        "flex": flex,
    }


def best_metric(pattern):
    metrics = []
    for path in glob.glob(pattern):
        metric = gsm8k_metric(path)
        if metric:
            metrics.append(metric)
    if not metrics:
        return None
    return sorted(metrics, key=lambda m: (m["n"], os.path.getmtime(m["path"])))[-1]


def sample_row(tag):
    data = load_json("experiments/phase19/gsm8k_sample_analysis.json") or {}
    for row in data.get("rows", []):
        if row.get("tag") == tag:
            return row
    return {}


def add(checks, status, name, detail):
    checks.append({"status": status, "name": name, "detail": detail})


def collect():
    checks = []

    for path in REQUIRED_ARTIFACTS:
        if Path(path).exists():
            add(checks, "pass", f"artifact:{path}", "present")
        else:
            add(checks, "fail", f"artifact:{path}", "missing")

    dense = best_metric("experiments/phase19/gsm8k_eval_baseline_full*.json")
    s20 = best_metric("experiments/phase19/gsm8k_eval_mask_s20_full*.json")
    s30 = best_metric("experiments/phase19/gsm8k_eval_mask_s30_full*.json")
    s40 = best_metric("experiments/phase19/gsm8k_eval_mask_s40_full*.json")
    phase23_meta = load_json("experiments/phase23_global_unstructured/Mistral-7B-v0.1_mask_pruning_global_s20.json")
    phase23 = best_metric("experiments/phase23_global_unstructured/gsm8k_eval_global_s20_full*.json")

    if dense and dense["n"] >= 1000:
        add(checks, "pass", "dense_full_gsm8k", f"n={dense['n']}, strict={dense['strict'] * 100:.2f}%")
    else:
        add(checks, "fail", "dense_full_gsm8k", "missing full dense baseline")

    if s20 and s20["n"] >= 1000:
        add(checks, "pass", "s20_full_gsm8k", f"n={s20['n']}, strict={s20['strict'] * 100:.2f}%")
    else:
        add(checks, "fail", "s20_full_gsm8k", "missing full 20% mask result")

    if dense and s20:
        delta = (s20["strict"] - dense["strict"]) * 100
        status = "pass" if delta >= -2.0 else "warn"
        add(checks, status, "s20_near_dense", f"delta={delta:+.2f} pt")

    if phase23_meta and phase23_meta.get("linear_sparsity") is not None:
        add(
            checks,
            "pass",
            "phase23_global_metadata",
            f"actual_sparsity={phase23_meta['linear_sparsity'] * 100:.2f}%, C4={phase23_meta.get('masked_c4_ppl', 0):.3f}",
        )
    else:
        add(checks, "fail", "phase23_global_metadata", "missing global 20% metadata")

    if phase23 and phase23["n"] >= 1000:
        add(checks, "pass", "phase23_global_full_gsm8k", f"n={phase23['n']}, strict={phase23['strict'] * 100:.2f}%")
    else:
        add(checks, "fail", "phase23_global_full_gsm8k", "missing full global 20% GSM8K result")

    if dense and phase23:
        delta = (phase23["strict"] - dense["strict"]) * 100
        add(checks, "pass", "phase23_global_vs_dense", f"delta={delta:+.2f} pt")

    dense_sample = sample_row("dense")
    s20_sample = sample_row("s20")
    word_delta = s20_sample.get("delta_avg_words_vs_dense_full")
    if word_delta is None:
        add(checks, "warn", "sample_length_sanity", "missing full-run sample-length delta")
    else:
        status = "pass" if abs(word_delta) <= 2.0 else "warn"
        add(checks, status, "sample_length_sanity", f"s20 avg-word delta={word_delta:+.2f}")

    marker_delta = None
    if dense_sample.get("final_marker_rate") is not None and s20_sample.get("final_marker_rate") is not None:
        marker_delta = (s20_sample["final_marker_rate"] - dense_sample["final_marker_rate"]) * 100
        status = "pass" if marker_delta >= -3.0 else "warn"
        add(checks, status, "final_marker_sanity", f"delta={marker_delta:+.1f} pt")
    else:
        add(checks, "warn", "final_marker_sanity", "missing final-answer-marker rates")

    if s30 and s30["n"] >= 1000:
        add(checks, "pass", "s30_full_gsm8k", f"n={s30['n']}, strict={s30['strict'] * 100:.2f}%")
    else:
        add(checks, "warn", "s30_full_gsm8k", "pending; full run in tmux")

    if s40 and s40["n"] >= 1000:
        add(checks, "pass", "s40_full_gsm8k", f"n={s40['n']}, strict={s40['strict'] * 100:.2f}%")
    else:
        add(checks, "warn", "s40_full_gsm8k", "pending; full run in tmux")

    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    state = "blocked" if failures else ("partial_positive_control" if warnings else "workshop_ready")
    return {"state": state, "checks": checks}


def print_markdown(report):
    print(f"Phase 19 readiness: {report['state']}")
    print()
    print("| status | check | detail |")
    print("| --- | --- | --- |")
    for check in report["checks"]:
        print(f"| {check['status']} | {check['name']} | {check['detail']} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_markdown(report)


if __name__ == "__main__":
    main()
