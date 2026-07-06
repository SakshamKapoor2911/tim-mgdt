"""Summarize Phase 21 auxiliary reasoning evaluations.

Phase 21 is intentionally lightweight: it reuses dense and Phase 19 s20
checkpoints on a small set of non-primary reasoning tasks. The goal is to test
whether the 20% mask result looks robust beyond the exact GSM8K full run.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path


PREFERRED_METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
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
    match = re.match(r"eval_([^_]+)_(.+)\.json$", Path(path).name)
    if not match:
        return None, ""
    return match.group(1), match.group(2)


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
            rows.append(
                {
                    "model_tag": model_tag,
                    "task": task,
                    "run_label": run_label,
                    "n": sample_count(data, task),
                    "metric": metric_name,
                    "value": metric_value,
                    "path": path,
                }
            )

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
    headers = ["run", "model", "task", "n", "metric", "value", "delta vs dense pt"]
    keys = ["run_label", "model_tag", "task", "n", "metric", "value", "delta_vs_dense_pt"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["rows"]:
        values = []
        for key in keys:
            value = row.get(key)
            if key == "value":
                values.append(pct(value))
            elif key == "delta_vs_dense_pt":
                values.append("" if value is None else f"{value:+.2f}")
            else:
                values.append(fmt(value))
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
