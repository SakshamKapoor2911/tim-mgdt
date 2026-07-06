"""Print a compact Phase 19 run and result status snapshot."""

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path


RUN_LOG_PATTERNS = {
    "s30_full": "experiments/phase19/run_logs/gsm8k_s30_full_*.log",
    "s40_full": "experiments/phase19/run_logs/gsm8k_s40_full_*.log",
}


def run(cmd):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return f"unavailable ({exc})"


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
        "n": samples.get("effective", gsm8k.get("sample_len")),
        "strict": gsm8k.get("exact_match,strict-match"),
        "flex": gsm8k.get("exact_match,flexible-extract"),
    }


def best_metric(pattern):
    candidates = []
    for path in glob.glob(pattern):
        metric = gsm8k_metric(path)
        if metric and metric.get("strict") is not None:
            metric["path"] = path
            candidates.append(metric)
    if not candidates:
        return None
    return sorted(candidates, key=lambda m: (m.get("n") or 0, os.path.getmtime(m["path"])))[-1]


def latest_path(pattern):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    return paths[-1] if paths else None


def parse_progress(path):
    if not path:
        return "missing log"
    log_path = Path(path)
    if not log_path.exists():
        return "missing log"
    text = log_path.read_text(errors="replace")
    matches = re.findall(
        r"Running generate_until requests:\s+([0-9]+)%.*?\|\s*([0-9]+)/1319\s*\[([^\]]+)\]",
        text,
    )
    if matches:
        pct, done, timing = matches[-1]
        return f"{done}/1319 ({pct}%), {timing}"
    if "Running generate_until requests" in text:
        return "generation started"
    if "Initializing hf model" in text:
        return "initializing model"
    return "no progress marker"


def latest_progress_in_log(path):
    log_path = Path(path)
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = re.findall(
        r"Running generate_until requests:\s+([0-9]+)%.*?\|\s*([0-9]+)/1319\s*\[([^\]]+)\]",
        text,
    )
    if not matches:
        return None
    pct, done, timing = matches[-1]
    return f"{done}/1319 ({pct}%), {timing}"


def print_log_tail(label, pattern, lines=3):
    latest = latest_path(pattern)
    if not latest:
        return
    tail = Path(latest).read_text(errors="replace").splitlines()[-lines:]
    print(f"{label}: {latest}")
    progress = latest_progress_in_log(latest)
    if progress:
        print(f"  progress: {progress}")
    for line in tail:
        print(f"  {line}")


def reference_for(metric, dense_full, dense_smoke):
    if not metric:
        return None, ""
    n = metric.get("n") or 0
    if n > 100 and dense_full:
        return dense_full, "full"
    if n <= 100 and dense_smoke:
        return dense_smoke, "smoke"
    if dense_full:
        return dense_full, "full"
    return dense_smoke, "smoke"


def structured_metric(path):
    data = load_json(path) or {}
    tag = Path(path).stem.split("_")[-1]
    return {
        "tag": tag,
        "target": data.get("target_structured_sparsity"),
        "actual": data.get("mlp_param_sparsity"),
        "c4": data.get("pruned_c4_ppl"),
        "tps": data.get("pruned_tps"),
    }


def percent_text(value):
    return "" if value is None else f"{value * 100:.2f}%"


def number_text(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def pct_number_text(value):
    return "" if value is None else f"{value * 100:.2f}"


def print_result_table():
    dense = best_metric("experiments/phase19/gsm8k_eval_baseline_full*.json")
    dense_smoke = best_metric("experiments/phase18b/gsm8k_eval_baseline*.json")
    print("\nResults:")
    print("tag     n       strict   flex     delta      ref")
    print("-----   -----   ------   ------   --------   -----")

    def row(tag, metric):
        if not metric:
            print(f"{tag:<7} pending")
            return
        strict = metric.get("strict")
        flex = metric.get("flex")
        ref, ref_name = reference_for(metric, dense, dense_smoke)
        ref_strict = ref.get("strict") if ref else None
        delta = None if strict is None or ref_strict is None else (strict - ref_strict) * 100
        delta_text = "" if delta is None else f"{delta:+.2f} pt"
        print(
            f"{tag:<7} {str(metric.get('n')):<7} "
            f"{strict * 100:>6.2f}   {flex * 100:>6.2f}   {delta_text:<8}   {ref_name}"
        )

    row("dense", dense)
    for tag in ("s20", "s30", "s40", "s25", "s35"):
        row(tag, best_metric(f"experiments/phase19/gsm8k_eval_mask_{tag}*.json"))


def print_phase20_table():
    result_paths = sorted(glob.glob("experiments/phase20_mask_guided/Mistral-7B-v0.1_mask_guided_width_*.json"))
    if not result_paths:
        print("\nPhase 20:")
        print("mask-guided structured sweep pending")
        return

    dense_smoke = best_metric("experiments/phase18b/gsm8k_eval_baseline*.json")
    dense_strict = dense_smoke.get("strict") if dense_smoke else None
    print("\nPhase 20 mask-guided structured results:")
    print("tag     target   actual   n       strict   delta      C4 PPL   tok/s")
    print("-----   ------   ------   -----   ------   --------   ------   -----")
    for path in result_paths:
        meta = structured_metric(path)
        metric = best_metric(f"experiments/phase20_mask_guided/gsm8k_eval_mask_guided_*_{meta['tag']}.json")
        strict = metric.get("strict") if metric else None
        delta = None if strict is None or dense_strict is None else (strict - dense_strict) * 100
        delta_text = "" if delta is None else f"{delta:+.2f} pt"
        print(
            f"{meta['tag']:<7} "
            f"{percent_text(meta['target']):>6}   "
            f"{percent_text(meta['actual']):>6}   "
            f"{str(metric.get('n') if metric else ''):<7} "
            f"{pct_number_text(strict):>6}   "
            f"{delta_text:<8}   "
            f"{number_text(meta['c4']):>6}   "
            f"{number_text(meta['tps'], 2):>5}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-system", action="store_true", help="Skip tmux and nvidia-smi probes.")
    args = parser.parse_args()

    if not args.no_system:
        print("tmux:")
        print(run(["tmux", "list-sessions"]))
        print("\nGPU:")
        print(
            run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
        )

    print("\nProgress:")
    for name, pattern in RUN_LOG_PATTERNS.items():
        path = latest_path(pattern)
        print(f"{name}: {parse_progress(path)}")

    print_log_tail("adaptive", "experiments/phase19/run_logs/adaptive_boundary_*.log")
    print_log_tail("phase20", "experiments/phase20_mask_guided/run_logs/phase20_queue_*.log")
    print_log_tail("publish", "experiments/publish_logs/publish_queue_*.log")

    print_result_table()
    print_phase20_table()


if __name__ == "__main__":
    main()
