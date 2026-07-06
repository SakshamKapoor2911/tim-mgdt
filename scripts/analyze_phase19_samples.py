"""Analyze logged Phase 19 GSM8K generations.

This reads lm-eval's `samples_gsm8k_*.jsonl` files and summarizes answer-chain
lengths next to the official strict/flexible metrics. It is intentionally
CPU-only and can be rerun after each full eval lands.
"""

import argparse
import glob
import json
import os
import re
import statistics
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


def tag_for_result(path):
    name = Path(path).stem
    if name.startswith("gsm8k_eval_baseline_full_"):
        return "dense"
    match = re.match(r"gsm8k_eval_mask_(s[0-9]+)", name)
    if match:
        return match.group(1)
    return name.replace("gsm8k_eval_", "")


def sample_path_for_result(path):
    name = Path(path).stem
    match = re.match(r"gsm8k_eval_(?:baseline_full|mask_s[0-9]+(?:_full)?)_(.+)", name)
    if not match:
        return None
    return str(Path(path).with_name(f"samples_gsm8k_{match.group(1)}.jsonl"))


def best_results(exp_dir):
    by_tag = {}
    for path in glob.glob(os.path.join(exp_dir, "gsm8k_eval_*.json")):
        metric = gsm8k_metric(path)
        if not metric:
            continue
        tag = tag_for_result(path)
        old = by_tag.get(tag)
        key = (metric.get("n") or 0, os.path.getmtime(path))
        old_key = ((old or {}).get("n") or 0, os.path.getmtime(old["path"])) if old else None
        if old is None or key > old_key:
            by_tag[tag] = metric
    return dict(sorted(by_tag.items(), key=lambda item: (0 if item[0] == "dense" else 1, item[0])))


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


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def round_or_none(value, digits=3):
    return None if value is None else round(value, digits)


def analyze_samples(path):
    if not path or not os.path.exists(path):
        return {
            "sample_path": path or "",
            "sample_rows": 0,
            "strict_rows": 0,
            "has_samples": False,
        }

    word_counts = []
    char_counts = []
    line_counts = []
    correct_word_counts = []
    incorrect_word_counts = []
    marker_count = 0
    sample_rows = 0
    strict_rows = 0

    with open(path) as f:
        for line in f:
            sample_rows += 1
            row = json.loads(line)
            if row.get("filter") != "strict-match":
                continue
            strict_rows += 1
            response = first_response(row)
            words = len(re.findall(r"\S+", response))
            chars = len(response)
            lines = 0 if response == "" else response.count("\n") + 1

            word_counts.append(words)
            char_counts.append(chars)
            line_counts.append(lines)
            if "####" in response:
                marker_count += 1
            if row.get("exact_match") == 1.0:
                correct_word_counts.append(words)
            else:
                incorrect_word_counts.append(words)

    return {
        "sample_path": path,
        "sample_rows": sample_rows,
        "strict_rows": strict_rows,
        "has_samples": strict_rows > 0,
        "avg_response_words": round_or_none(mean(word_counts), 3),
        "median_response_words": round_or_none(statistics.median(word_counts), 3) if word_counts else None,
        "p90_response_words": percentile(word_counts, 0.90),
        "avg_response_chars": round_or_none(mean(char_counts), 3),
        "median_response_lines": round_or_none(statistics.median(line_counts), 3) if line_counts else None,
        "final_marker_rate": round_or_none(marker_count / strict_rows, 3) if strict_rows else None,
        "avg_words_correct": round_or_none(mean(correct_word_counts), 3),
        "avg_words_incorrect": round_or_none(mean(incorrect_word_counts), 3),
    }


def enrich_row(tag, metric, exp_dir):
    sample_path = sample_path_for_result(metric["path"])
    row = {
        "tag": tag,
        "scale": "full" if (metric.get("n") or 0) > 100 else "smoke",
        "n": metric.get("n"),
        "strict": metric.get("strict"),
        "flex": metric.get("flex"),
        "metric_path": metric.get("path"),
    }

    if tag == "dense":
        mask_paths = sorted(glob.glob(os.path.join(exp_dir, "Mistral-7B-v0.1_mask_pruning_s*.json")))
        first_mask = load_json(mask_paths[0]) if mask_paths else {}
        row.update({"target_sparsity": 0.0, "linear_sparsity": 0.0, "c4_ppl": (first_mask or {}).get("baseline_c4_ppl")})
    else:
        mask_meta = load_json(os.path.join(exp_dir, f"Mistral-7B-v0.1_mask_pruning_{tag}.json")) or {}
        row.update(
            {
                "target_sparsity": mask_meta.get("target_sparsity"),
                "linear_sparsity": mask_meta.get("linear_sparsity"),
                "c4_ppl": mask_meta.get("masked_c4_ppl"),
            }
        )

    row.update(analyze_samples(sample_path))
    return row


def collect(exp_dir):
    rows = [enrich_row(tag, metric, exp_dir) for tag, metric in best_results(exp_dir).items()]
    dense = next((row for row in rows if row["tag"] == "dense"), None)
    dense_full = dense if dense and dense.get("scale") == "full" else None
    dense_strict = dense_full.get("strict") if dense_full else None
    dense_words = dense_full.get("avg_response_words") if dense_full else None

    for row in rows:
        if row.get("scale") != "full":
            row["delta_strict_vs_dense_full_pt"] = None
            row["delta_avg_words_vs_dense_full"] = None
            continue
        row["delta_strict_vs_dense_full_pt"] = (
            None if dense_strict is None else round((row.get("strict") - dense_strict) * 100, 3)
        )
        row["delta_avg_words_vs_dense_full"] = (
            None
            if dense_words is None or row.get("avg_response_words") is None
            else round(row.get("avg_response_words") - dense_words, 3)
        )

    return {
        "schema": "phase19_gsm8k_sample_analysis_v1",
        "exp_dir": exp_dir,
        "rows": rows,
        "notes": [
            "Generation-length statistics use lm-eval strict-match sample rows only.",
            "Accuracy fields come from the official lm-eval aggregate JSON.",
            "Full-run deltas are only computed against the full dense baseline.",
        ],
    }


def pct(value):
    return "" if value is None else f"{value * 100:.2f}"


def format_number(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def print_markdown(summary):
    headers = [
        "tag",
        "scale",
        "n",
        "strict EM %",
        "flex EM %",
        "C4 PPL",
        "avg words",
        "median words",
        "p90 words",
        "final marker %",
        "delta strict full pt",
        "delta words full",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary["rows"]:
        values = [
            row.get("tag"),
            row.get("scale"),
            row.get("n"),
            pct(row.get("strict")),
            pct(row.get("flex")),
            format_number(row.get("c4_ppl"), 3),
            format_number(row.get("avg_response_words"), 2),
            format_number(row.get("median_response_words"), 2),
            format_number(row.get("p90_response_words"), 2),
            pct(row.get("final_marker_rate")),
            format_number(row.get("delta_strict_vs_dense_full_pt"), 2),
            format_number(row.get("delta_avg_words_vs_dense_full"), 2),
        ]
        print("| " + " | ".join(str(value) for value in values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="experiments/phase19")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of Markdown.")
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
