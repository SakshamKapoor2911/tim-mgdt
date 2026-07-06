"""Summarize Phase 22 cross-model dense-vs-mask smoke checks."""

import argparse
import glob
import json
import os
import re
from pathlib import Path


PREFERRED_METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,get-answer",
    "exact_match,none",
    "acc,none",
    "acc_norm,none",
]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def repo_relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def parse_eval_name(path):
    name = Path(path).name
    match = re.match(
        r"eval_(.+?)_(dense|s[0-9]+)_([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2})(?:_.+)?\.json$",
        name,
    )
    if not match:
        return None, None, ""
    return match.group(1), match.group(2), match.group(3)


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


def mask_metadata(exp_dir, model_tag, mask_tag):
    candidates = sorted(glob.glob(os.path.join(exp_dir, f"*_mask_pruning_{model_tag}_{mask_tag}.json")))
    if not candidates:
        return {}
    return load_json(candidates[-1]) or {}


def baseline_metadata(exp_dir, model_tag):
    candidates = sorted(glob.glob(os.path.join(exp_dir, f"*_mask_pruning_{model_tag}_s*.json")))
    if not candidates:
        return {}
    return load_json(candidates[-1]) or {}


def collect(exp_dir, model_tag=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "eval_*.json"))):
        parsed_model_tag, state, run_label = parse_eval_name(path)
        if not parsed_model_tag or (model_tag and parsed_model_tag != model_tag):
            continue
        data = load_json(path)
        if not data:
            continue
        meta = baseline_metadata(exp_dir, parsed_model_tag) if state == "dense" else mask_metadata(exp_dir, parsed_model_tag, state)
        for task, result in sorted(data.get("results", {}).items()):
            metric_name, metric_value = metric_for(result)
            rows.append(
                {
                    "model_tag": parsed_model_tag,
                    "state": state,
                    "task": task,
                    "run_label": run_label,
                    "n": sample_count(data, task),
                    "metric": metric_name,
                    "value": metric_value,
                    "target_sparsity": 0.0 if state == "dense" else meta.get("target_sparsity"),
                    "linear_sparsity": 0.0 if state == "dense" else meta.get("linear_sparsity"),
                    "c4_ppl": meta.get("baseline_c4_ppl") if state == "dense" else meta.get("masked_c4_ppl"),
                    "path": repo_relative(path),
                    "_mtime": Path(path).stat().st_mtime,
                }
            )

    deduped = {}
    for row in rows:
        key = (
            row["model_tag"],
            row["state"],
            row["task"],
            row["run_label"],
            row["metric"],
        )
        previous = deduped.get(key)
        if previous is None or row["_mtime"] > previous["_mtime"]:
            deduped[key] = row
    rows = sorted(
        deduped.values(),
        key=lambda row: (
            row["run_label"],
            row["model_tag"],
            row["task"],
            0 if row["state"] == "dense" else 1,
            row["state"],
        ),
    )
    for row in rows:
        row.pop("_mtime", None)

    dense_by_key = {
        (row["model_tag"], row["task"], row["run_label"]): row
        for row in rows
        if row["state"] == "dense" and row["value"] is not None
    }
    for row in rows:
        dense = dense_by_key.get((row["model_tag"], row["task"], row["run_label"]))
        if not dense or row["value"] is None:
            row["delta_vs_dense_pt"] = None
        else:
            row["delta_vs_dense_pt"] = round((row["value"] - dense["value"]) * 100, 3)

    return {
        "schema": "phase22_cross_model_summary_v1",
        "exp_dir": repo_relative(exp_dir),
        "rows": rows,
        "notes": [
            "Phase 22 is a bounded cross-model smoke check, not a full benchmark.",
            "Deltas compare the same task and run label against that model's dense state.",
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
        "state",
        "task",
        "n",
        "metric",
        "value",
        "delta vs dense pt",
        "C4 PPL",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["rows"]:
        print(
            "| "
            + " | ".join(
                [
                    fmt(row.get("run_label")),
                    fmt(row.get("model_tag")),
                    fmt(row.get("state")),
                    fmt(row.get("task")),
                    fmt(row.get("n")),
                    fmt(row.get("metric")),
                    pct(row.get("value")),
                    "" if row.get("delta_vs_dense_pt") is None else f"{row['delta_vs_dense_pt']:+.2f}",
                    fmt(row.get("c4_ppl")),
                ]
            )
            + " |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/phase22_cross_model")
    parser.add_argument("--model-tag", help="Optional model tag filter.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    summary = collect(args.exp_dir, args.model_tag)
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
