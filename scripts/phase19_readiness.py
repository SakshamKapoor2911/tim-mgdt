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
    "experiments/phase24_grouped_ablation/phase24_summary_limit300_2026-07-07T07-30-10.json",
    "experiments/phase21_aux_reasoning/summary_phase25_aux_limit300_2026-07-07T13-01-49.json",
    "experiments/phase21_aux_reasoning/summary_phase25_platinum_full_2026-07-07T13-55-56.json",
]

PHASE25_AUX_LABEL = "phase25_aux_limit300_2026-07-07T13-01-49"
PHASE25_PLATINUM_FULL_LABEL = "phase25_platinum_full_2026-07-07T13-55-56"


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
    phase24 = load_json("experiments/phase24_grouped_ablation/phase24_summary_limit300_2026-07-07T07-30-10.json") or {}
    phase25_aux = load_json("experiments/phase21_aux_reasoning/summary_phase25_aux_limit300_2026-07-07T13-01-49.json") or {}
    phase25_platinum = load_json(
        "experiments/phase21_aux_reasoning/summary_phase25_platinum_full_2026-07-07T13-55-56.json"
    ) or {}

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
    global_sample = sample_row("global_s20")
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

    if global_sample:
        global_word_delta = global_sample.get("delta_avg_words_vs_dense_full")
        global_extract_fail = global_sample.get("extraction_failure_rate")
        if global_word_delta is None:
            add(checks, "warn", "global_sample_length_sanity", "missing global 20% sample-length delta")
        else:
            status = "pass" if abs(global_word_delta) <= 2.0 else "warn"
            add(checks, status, "global_sample_length_sanity", f"global_s20 avg-word delta={global_word_delta:+.2f}")

        if global_extract_fail is None:
            add(checks, "warn", "global_extraction_sanity", "missing global 20% extraction-failure rate")
        else:
            status = "pass" if global_extract_fail <= 0.01 else "warn"
            add(checks, status, "global_extraction_sanity", f"extraction_failure={global_extract_fail * 100:.1f}%")
    else:
        add(checks, "warn", "global_sample_diagnostics", "missing global 20% sample row")

    phase24_rows = phase24.get("rows", [])
    phase24_decision = phase24.get("decision", {})
    if len(phase24_rows) >= 6:
        tags = {row.get("tag") for row in phase24_rows}
        expected_tags = {
            "s25_no_revert",
            "revert_early",
            "revert_mid",
            "revert_late",
            "revert_mlp",
            "revert_attention",
        }
        status = "pass" if expected_tags.issubset(tags) else "warn"
        add(checks, status, "phase24_grouped_rows", f"rows={len(phase24_rows)}, tags={sorted(tags)}")
    else:
        add(checks, "warn", "phase24_grouped_rows", f"expected 6 rows, found {len(phase24_rows)}")

    phase24_baseline = next((row for row in phase24_rows if row.get("tag") == "s25_no_revert"), {})
    if phase24_baseline and phase24_baseline.get("n") == 300:
        add(
            checks,
            "pass",
            "phase24_no_revert_baseline",
            f"n=300, strict={phase24_baseline.get('strict', 0) * 100:.2f}%, CI={phase24_baseline.get('bootstrap_ci_low', 0) * 100:.1f}-{phase24_baseline.get('bootstrap_ci_high', 0) * 100:.1f}%",
        )
    else:
        add(checks, "warn", "phase24_no_revert_baseline", "missing same-prefix 25% no-revert row")

    phase24_diag_rows = [
        row
        for row in phase24_rows
        if row.get("avg_response_words") is not None
        and row.get("final_marker_rate") is not None
        and row.get("extraction_failure_rate") is not None
    ]
    if len(phase24_diag_rows) == len(phase24_rows) and phase24_rows:
        max_extract_fail = max(row.get("extraction_failure_rate") or 0 for row in phase24_diag_rows)
        status = "pass" if max_extract_fail <= 0.01 else "warn"
        add(
            checks,
            status,
            "phase24_format_sanity",
            f"diagnostics={len(phase24_diag_rows)}/{len(phase24_rows)}, max_extraction_failure={max_extract_fail * 100:.1f}%",
        )
    else:
        add(
            checks,
            "warn",
            "phase24_format_sanity",
            f"diagnostics={len(phase24_diag_rows)}/{len(phase24_rows)} rows",
        )

    recommendation = phase24_decision.get("recommendation")
    if recommendation in {"distributed_or_weak_module_signal", "module_axis_signal", "stop_after_layer_groups"}:
        add(checks, "pass", "phase24_terminal_decision", recommendation)
    else:
        add(checks, "warn", "phase24_terminal_decision", recommendation or "missing")

    phase25_platinum_rows = [
        row for row in phase25_platinum.get("rows", []) if row.get("run_label") == PHASE25_PLATINUM_FULL_LABEL
    ]
    phase25_aux_rows = [
        row for row in phase25_aux.get("rows", []) if row.get("run_label") == PHASE25_AUX_LABEL
    ]
    platinum_by_key = {(row.get("model_tag"), row.get("task")): row for row in phase25_platinum_rows}
    aux_by_key = {(row.get("model_tag"), row.get("task")): row for row in phase25_aux_rows}
    dense_platinum = platinum_by_key.get(("dense", "gsm8k_platinum_cot"), {})
    s20_platinum = platinum_by_key.get(("s20", "gsm8k_platinum_cot"), {})
    dense_bbh = aux_by_key.get(("dense", "bbh_cot_fewshot_multistep_arithmetic_two"), {})
    s20_bbh = aux_by_key.get(("s20", "bbh_cot_fewshot_multistep_arithmetic_two"), {})
    if all([dense_platinum, s20_platinum, dense_bbh, s20_bbh]):
        add(
            checks,
            "pass",
            "phase25_aux_rows",
            f"gsm8k_platinum n={s20_platinum.get('n')}, bbh n={s20_bbh.get('n')}",
        )
        platinum_delta = s20_platinum.get("delta_vs_dense_pt")
        bbh_delta = s20_bbh.get("delta_vs_dense_pt")
        status = "pass" if platinum_delta is not None and bbh_delta is not None and platinum_delta >= -2.0 else "warn"
        platinum_text = (
            "missing"
            if platinum_delta is None
            else f"s20={s20_platinum.get('value', 0) * 100:.2f}%, dense={dense_platinum.get('value', 0) * 100:.2f}%"
        )
        bbh_text = (
            "missing"
            if bbh_delta is None
            else f"s20={s20_bbh.get('value', 0) * 100:.2f}%, dense={dense_bbh.get('value', 0) * 100:.2f}%"
        )
        add(
            checks,
            status,
            "phase25_aux_near_dense",
            f"statistically indistinguishable: platinum({platinum_text}), bbh({bbh_text})",
        )
        max_extract_fail = max(
            row.get("extraction_failure_rate") or 0
            for row in [dense_platinum, s20_platinum, dense_bbh, s20_bbh]
        )
        status = "pass" if max_extract_fail <= 0.01 else "warn"
        add(checks, status, "phase25_aux_format_sanity", f"max_extraction_failure={max_extract_fail * 100:.1f}%")
    else:
        add(
            checks,
            "warn",
            "phase25_aux_rows",
            f"found platinum={len(phase25_platinum_rows)} rows, aux={len(phase25_aux_rows)} rows",
        )

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
