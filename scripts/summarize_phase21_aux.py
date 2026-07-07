"""Summarize Phase 21 auxiliary reasoning evaluations.

Phase 21 is intentionally lightweight: it reuses dense and Phase 19 s20
checkpoints on a small set of non-primary reasoning tasks. The goal is to test
whether the 20% mask result looks robust beyond the exact GSM8K full run.
"""

import argparse
import glob
import json
import os
import random
import re
import statistics
from pathlib import Path


PREFERRED_METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,get-answer",
    "exact_match,none",
    "acc,none",
    "acc_norm,none",
    "alias",
]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_eval_name(path):
    stem = Path(path).stem
    match = re.match(r"eval_([^_]+)_(.+)$", stem)
    if not match:
        return None, ""
    model_tag, run_label = match.groups()
    run_label = re.sub(
        r"_[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}(?:\.[0-9]+)?$",
        "",
        run_label,
    )
    return model_tag, run_label


def metric_for(task_result):
    for key in PREFERRED_METRICS:
        value = task_result.get(key)
        if isinstance(value, (int, float)):
            return key, value
    for key, value in sorted(task_result.items()):
        if isinstance(value, (int, float)):
            return key, value
    return "", None


def sample_count(data, task):
    samples = data.get("n-samples", {}).get(task, {})
    if isinstance(samples, dict):
        return samples.get("effective", samples.get("original"))
    return None


def sample_path_for_result(path, task):
    stem = Path(path).stem
    match = re.search(
        r"_([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}(?:\.[0-9]+)?)$",
        stem,
    )
    if match:
        candidate = Path(path).parent / f"samples_{task}_{match.group(1)}.jsonl"
        if candidate.exists():
            return str(candidate)
    candidates = sorted(Path(path).parent.glob(f"samples_{task}_*.jsonl"), key=os.path.getmtime)
    return str(candidates[-1]) if candidates else None


def filter_name(metric):
    if "," not in metric:
        return None
    return metric.split(",", 1)[1]


def strict_values(sample_path, metric):
    values = []
    target_filter = filter_name(metric)
    if not sample_path or not Path(sample_path).exists() or not target_filter:
        return values
    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("filter") != target_filter:
                continue
            if row.get("exact_match") is None:
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


def first_response(row):
    resps = row.get("resps") or []
    if not resps:
        return ""
    first = resps[0]
    if isinstance(first, list):
        return str(first[0]) if first else ""
    return str(first)


def sample_diagnostics(sample_path, metric):
    diagnostics = {
        "avg_response_words": None,
        "median_response_words": None,
        "answer_marker_rate": None,
        "extraction_failure_rate": None,
    }
    target_filter = filter_name(metric)
    if not sample_path or not Path(sample_path).exists() or not target_filter:
        return diagnostics

    word_counts = []
    marker_count = 0
    extraction_failure_count = 0
    row_count = 0
    with open(sample_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("filter") != target_filter:
                continue
            row_count += 1
            response = first_response(row)
            word_counts.append(len(re.findall(r"\S+", response)))
            if "answer is" in response.lower() or "####" in response:
                marker_count += 1
            filtered = row.get("filtered_resps") or []
            if not filtered or filtered[0] in ("", None):
                extraction_failure_count += 1

    diagnostics.update(
        {
            "avg_response_words": round(sum(word_counts) / len(word_counts), 3) if word_counts else None,
            "median_response_words": round(statistics.median(word_counts), 3) if word_counts else None,
            "answer_marker_rate": round(marker_count / row_count, 3) if row_count else None,
            "extraction_failure_rate": round(extraction_failure_count / row_count, 3) if row_count else None,
        }
    )
    return diagnostics


def collect(exp_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "eval_*.json"))):
        data = load_json(path)
        if not data:
            continue
        model_tag, run_label = parse_eval_name(path)
        if not model_tag:
            continue
        for task, result in sorted(data.get("results", {}).items()):
            metric_name, metric_value = metric_for(result)
            sample_path = sample_path_for_result(path, task)
            values = strict_values(sample_path, metric_name)
            ci_low, ci_high = bootstrap_ci(values)
            row = {
                "model_tag": model_tag,
                "task": task,
                "run_label": run_label,
                "n": sample_count(data, task),
                "metric": metric_name,
                "value": metric_value,
                "path": path,
                "sample_path": sample_path,
                "sample_n": len(values),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
            row.update(sample_diagnostics(sample_path, metric_name))
            rows.append(row)

    dense_by_run_task = {
        (row["run_label"], row["task"]): row
        for row in rows
        if row["model_tag"] == "dense" and row["value"] is not None
    }
    for row in rows:
        dense = dense_by_run_task.get((row["run_label"], row["task"]))
        if not dense or row["value"] is None:
            row["delta_vs_dense_pt"] = None
            continue
        row["delta_vs_dense_pt"] = round((row["value"] - dense["value"]) * 100, 3)

    return {
        "schema": "phase21_aux_reasoning_summary_v1",
        "exp_dir": exp_dir,
        "rows": rows,
        "notes": [
            "Auxiliary evals are smoke-sized checks, not replacements for full GSM8K.",
            "Deltas compare rows with the same task and run label against dense.",
        ],
    }


def pct(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}" if -1.0 <= value <= 1.0 else f"{value:.3f}"


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(summary):
    headers = [
        "run",
        "model",
        "task",
        "n",
        "metric",
        "value",
        "95% CI",
        "delta vs dense pt",
        "avg words",
        "extract fail %",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["rows"]:
        ci = ""
        if row.get("bootstrap_ci_low") is not None and row.get("bootstrap_ci_high") is not None:
            ci = f"{row['bootstrap_ci_low'] * 100:.1f}-{row['bootstrap_ci_high'] * 100:.1f}"
        values = [
            fmt(row.get("run_label")),
            fmt(row.get("model_tag")),
            fmt(row.get("task")),
            fmt(row.get("n")),
            fmt(row.get("metric")),
            pct(row.get("value")),
            ci,
            "" if row.get("delta_vs_dense_pt") is None else f"{row['delta_vs_dense_pt']:+.2f}",
            fmt(row.get("avg_response_words")),
            pct(row.get("extraction_failure_rate")),
        ]
        print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/phase21_aux_reasoning")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    summary = collect(args.exp_dir)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_markdown(summary)


if __name__ == "__main__":
    main()
