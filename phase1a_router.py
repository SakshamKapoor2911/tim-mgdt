#!/usr/bin/env python3
"""
Phase 1a: train a lightweight rejection router from Phase 0 hidden states.

The Phase 0 collator writes X/y arrays but the JSONL carries prompt metadata.
This script keeps evaluation honest by splitting by prompt id, preventing tokens
from the same GSM8K prompt from leaking across train/validation/test splits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_DIR = Path("rejection_mapping_logs")
DATASET_NAME = "phase1a_router_dataset.npz"
JSONL_NAME = "rejection_logs_phase0.jsonl"
RESULTS_NAME = "phase1a_router_results.json"
MODEL_NAME = "phase1a_router_model.joblib"
PLOT_NAME = "phase1a_router_metrics.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a Phase 1a rejection router.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-prompts", type=int, default=7)
    parser.add_argument("--test-prompts", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ordered_logs_for_collated_array(logs: Sequence[Dict]) -> List[Dict]:
    return sorted(logs, key=lambda item: (int(item["prompt_id"]), int(item["token_index"])))


def load_dataset(output_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    dataset_path = output_dir / DATASET_NAME
    logs_path = output_dir / JSONL_NAME
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing collated dataset: {dataset_path}")
    if not logs_path.exists():
        raise FileNotFoundError(f"Missing rejection logs: {logs_path}")

    with np.load(dataset_path) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)

    logs = ordered_logs_for_collated_array(load_jsonl(logs_path))
    if len(logs) != len(y):
        raise ValueError(f"Log/data length mismatch: {len(logs)} logs vs {len(y)} labels")

    y_from_logs = np.asarray([1 if log["rejected"] else 0 for log in logs], dtype=np.int64)
    if not np.array_equal(y, y_from_logs):
        raise ValueError("Collated labels do not align with sorted JSONL logs")

    return X, y, logs


def split_prompt_ids(
    logs: Sequence[Dict],
    *,
    seed: int,
    val_prompts: int,
    test_prompts: int,
) -> Dict[str, List[int]]:
    prompt_ids = sorted({int(log["prompt_id"]) for log in logs})
    if val_prompts + test_prompts >= len(prompt_ids):
        raise ValueError("Validation + test prompt counts must leave at least one train prompt")

    rng = random.Random(seed)
    shuffled = prompt_ids[:]
    rng.shuffle(shuffled)
    test_ids = sorted(shuffled[:test_prompts])
    val_ids = sorted(shuffled[test_prompts : test_prompts + val_prompts])
    train_ids = sorted(shuffled[test_prompts + val_prompts :])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def indices_for_prompt_ids(logs: Sequence[Dict], prompt_ids: Sequence[int]) -> np.ndarray:
    prompt_set = set(prompt_ids)
    return np.asarray(
        [idx for idx, log in enumerate(logs) if int(log["prompt_id"]) in prompt_set],
        dtype=np.int64,
    )


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict:
    y_pred = (probabilities >= threshold).astype(np.int64)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()) if y_true.size else 0.0,
        "predicted_positive_rate": float(y_pred.mean()) if y_pred.size else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def choose_threshold_for_f1(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def train_with_validation(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    c_grid: Sequence[float],
    max_iter: int,
    seed: int,
) -> Tuple[Pipeline, Dict]:
    best = None

    for c_value in c_grid:
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        val_prob = model.predict_proba(X[val_idx])[:, 1]
        threshold, val_f1 = choose_threshold_for_f1(y[val_idx], val_prob)
        val_metrics = metric_dict(y[val_idx], val_prob, threshold)
        val_metrics["best_f1_from_curve"] = val_f1
        val_metrics["C"] = float(c_value)

        score = (
            val_metrics["f1"],
            val_metrics["average_precision"] if val_metrics["average_precision"] is not None else -1.0,
            -abs(val_metrics["predicted_positive_rate"] - val_metrics["positive_rate"]),
        )
        if best is None or score > best["score"]:
            best = {"model": model, "metrics": val_metrics, "threshold": threshold, "score": score}

    assert best is not None
    return best["model"], best


def prompt_label_summary(logs: Sequence[Dict], split_ids: Dict[str, List[int]]) -> Dict:
    summary = {}
    for split_name, prompt_ids in split_ids.items():
        idxs = indices_for_prompt_ids(logs, prompt_ids)
        labels = np.asarray([1 if logs[idx]["rejected"] else 0 for idx in idxs], dtype=np.int64)
        summary[split_name] = {
            "prompt_ids": prompt_ids,
            "tokens": int(labels.size),
            "rejections": int(labels.sum()),
            "rejection_rate": float(labels.mean()) if labels.size else 0.0,
        }
    return summary


def plot_metrics(y_test: np.ndarray, test_prob: np.ndarray, threshold: float, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        precision, recall, _ = precision_recall_curve(y_test, test_prob)
        axes[0].plot(fpr, tpr, color="#345995")
        axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        axes[0].set_title("ROC Curve")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].grid(alpha=0.25)

        axes[1].plot(recall, precision, color="#d95d39")
        axes[1].set_title("Precision-Recall Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].grid(alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "Single-class test split", ha="center", va="center")
        axes[1].text(0.5, 0.5, "Single-class test split", ha="center", va="center")

    axes[2].hist(test_prob[y_test == 0], bins=25, alpha=0.75, label="accepted", color="#345995")
    axes[2].hist(test_prob[y_test == 1], bins=25, alpha=0.75, label="rejected", color="#d95d39")
    axes[2].axvline(threshold, color="#111111", linestyle="--", linewidth=1, label="threshold")
    axes[2].set_title("Router Score Histogram")
    axes[2].set_xlabel("P(reject)")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Phase 1a Rejection Router Evaluation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    X, y, logs = load_dataset(args.output_dir)
    split_ids = split_prompt_ids(
        logs,
        seed=args.seed,
        val_prompts=args.val_prompts,
        test_prompts=args.test_prompts,
    )
    train_idx = indices_for_prompt_ids(logs, split_ids["train"])
    val_idx = indices_for_prompt_ids(logs, split_ids["val"])
    test_idx = indices_for_prompt_ids(logs, split_ids["test"])

    model, best = train_with_validation(
        X,
        y,
        train_idx,
        val_idx,
        args.c_grid,
        args.max_iter,
        args.seed,
    )
    threshold = float(best["threshold"])
    train_prob = model.predict_proba(X[train_idx])[:, 1]
    val_prob = model.predict_proba(X[val_idx])[:, 1]
    test_prob = model.predict_proba(X[test_idx])[:, 1]

    results = {
        "config": {
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "val_prompts": args.val_prompts,
            "test_prompts": args.test_prompts,
            "c_grid": list(args.c_grid),
            "max_iter": args.max_iter,
        },
        "dataset": {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "rejection_rate": float(y.mean()),
            "total_rejections": int(y.sum()),
        },
        "splits": prompt_label_summary(logs, split_ids),
        "selected": {
            "C": float(best["metrics"]["C"]),
            "threshold": threshold,
            "selection_metric": "validation_f1",
        },
        "train": metric_dict(y[train_idx], train_prob, threshold),
        "validation": best["metrics"],
        "test": metric_dict(y[test_idx], test_prob, threshold),
        "default_threshold_test": metric_dict(y[test_idx], test_prob, 0.5),
    }

    joblib.dump(model, args.output_dir / MODEL_NAME)
    with (args.output_dir / RESULTS_NAME).open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_metrics(y[test_idx], test_prob, threshold, args.output_dir / PLOT_NAME)

    print("Phase 1a router complete")
    print(f"  dataset: X{X.shape}, y{y.shape}, rejection_rate={y.mean():.4f}")
    print(f"  selected C: {results['selected']['C']}")
    print(f"  selected threshold: {threshold:.4f}")
    print(f"  val f1: {results['validation']['f1']:.4f}")
    print(f"  test f1: {results['test']['f1']:.4f}")
    print(f"  test roc_auc: {results['test']['roc_auc']}")
    print(f"  test average_precision: {results['test']['average_precision']}")
    print(f"  wrote: {args.output_dir / RESULTS_NAME}")
    print(f"  wrote: {args.output_dir / MODEL_NAME}")
    print(f"  wrote: {args.output_dir / PLOT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
