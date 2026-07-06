"""Print a compact Phase 19 run and result status snapshot."""

import argparse
import glob
import json
import os
import re
import subprocess
from pathlib import Path


RUN_LOGS = {
    "s30_full": "experiments/phase19/run_logs/gsm8k_s30_full_2026-07-06T12-05-47.log",
    "s40_full": "experiments/phase19/run_logs/gsm8k_s40_full_2026-07-06T12-05-47.log",
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


def parse_progress(path):
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


def print_result_table():
    dense = best_metric("experiments/phase19/gsm8k_eval_baseline_full*.json")
    dense_strict = dense.get("strict") if dense else None
    print("\nResults:")
    print("tag     n       strict   flex     delta_vs_dense")
    print("-----   -----   ------   ------   --------------")

    def row(tag, metric):
        if not metric:
            print(f"{tag:<7} pending")
            return
        strict = metric.get("strict")
        flex = metric.get("flex")
        delta = None if strict is None or dense_strict is None else (strict - dense_strict) * 100
        delta_text = "" if delta is None else f"{delta:+.2f} pt"
        print(
            f"{tag:<7} {str(metric.get('n')):<7} "
            f"{strict * 100:>6.2f}   {flex * 100:>6.2f}   {delta_text}"
        )

    row("dense", dense)
    for tag in ("s20", "s30", "s40", "s25", "s35"):
        row(tag, best_metric(f"experiments/phase19/gsm8k_eval_mask_{tag}*.json"))


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
    for name, path in RUN_LOGS.items():
        print(f"{name}: {parse_progress(path)}")

    adaptive_logs = sorted(glob.glob("experiments/phase19/run_logs/adaptive_boundary_*.log"))
    if adaptive_logs:
        latest = adaptive_logs[-1]
        tail = Path(latest).read_text(errors="replace").splitlines()[-3:]
        print(f"adaptive: {latest}")
        adaptive_progress = latest_progress_in_log(latest)
        if adaptive_progress:
            print(f"  progress: {adaptive_progress}")
        for line in tail:
            print(f"  {line}")

    print_result_table()


if __name__ == "__main__":
    main()
