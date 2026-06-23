#!/usr/bin/env python3
"""
Phase 0: Rejection Mapping for speculative decoding.

This script builds a hardware-calibrated dataset for a routing-driven
speculative decoding gate. It keeps the draft and target models on separate
GPUs, drafts K tokens iteratively, verifies them with one parallel target
forward pass, and logs each verified draft token with the draft hidden state
that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_OUTPUT_DIR = Path("rejection_mapping_logs")
JSONL_NAME = "rejection_logs_phase0.jsonl"
SUMMARY_NAME = "phase0_summary.json"
COLLATED_NAME = "phase1a_router_dataset.npz"
PLOT_NAME = "rejection_patterns.png"


@dataclass
class CandidateRecord:
    prompt_id: int
    step_index: int
    draft_step: int
    token_index: int
    draft_token_id: int
    hidden_state: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Phase 0 hidden-state/rejection labels for speculative decoding."
    )
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer model id. Defaults to target model.")
    parser.add_argument("--draft-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--draft-length", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--warmup-examples", type=int, default=10)
    parser.add_argument("--min-warmup-alpha", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--skip-collate", action="store_true", help="Do not save Phase 1a X/y arrays.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached HF files.")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dual_cuda(draft_device: str, target_device: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this run requires two CUDA GPUs.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Expected at least 2 CUDA GPUs, found {torch.cuda.device_count()}.")
    if torch.device(draft_device).type != "cuda" or torch.device(target_device).type != "cuda":
        raise RuntimeError("Draft and target devices must both be CUDA devices.")
    if torch.device(draft_device).index == torch.device(target_device).index:
        raise RuntimeError("Draft and target devices must be different GPUs.")


def print_gpu_resource_snapshot() -> None:
    print("GPU resource snapshot:")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        free_gib = free_bytes / (1024**3)
        total_gib = total_bytes / (1024**3)
        print(
            f"  cuda:{idx}: {props.name}, free {free_gib:.2f} GiB / {total_gib:.2f} GiB"
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains files. Pass --overwrite to replace Phase 0 artifacts."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def load_tokenizer(tokenizer_name: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_on_device(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    model.eval()
    model.config.use_cache = False
    return model


def load_models_and_tokenizer(args: argparse.Namespace):
    dtype = torch_dtype(args.dtype)
    tokenizer_name = args.tokenizer or args.target_model

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name, args.local_files_only)

    print(f"Loading draft model on {args.draft_device}: {args.draft_model}")
    draft_model = load_model_on_device(
        args.draft_model, args.draft_device, dtype, args.local_files_only
    )
    print(f"Draft: {count_parameters(draft_model):,} params")
    print(f"Draft first parameter device: {first_parameter_device(draft_model)}")

    print(f"Loading target model on {args.target_device}: {args.target_model}")
    target_model = load_model_on_device(
        args.target_model, args.target_device, dtype, args.local_files_only
    )
    print(f"Target: {count_parameters(target_model):,} params")
    print(f"Target first parameter device: {first_parameter_device(target_model)}")

    return draft_model, target_model, tokenizer


def load_gsm8k_examples(num_examples: int) -> List[Dict]:
    split = f"test[:{num_examples}]"
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row.get("answer", "")} for row in dataset]


def sync_devices(devices: Iterable[torch.device]) -> None:
    if not torch.cuda.is_available():
        return
    seen = set()
    for device in devices:
        if device.type != "cuda":
            continue
        idx = device.index
        if idx in seen:
            continue
        torch.cuda.synchronize(device)
        seen.add(idx)


def safe_decode(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def argmax_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def build_metrics() -> Dict[str, float]:
    return {
        "prompts": 0,
        "verified_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "total_time_sec": 0.0,
        "acceptance_rate": 0.0,
        "rejection_rate": 0.0,
        "tokens_per_sec": 0.0,
    }


def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    verified = metrics["verified_tokens"]
    elapsed = metrics["total_time_sec"]
    metrics["acceptance_rate"] = metrics["accepted_tokens"] / verified if verified else 0.0
    metrics["rejection_rate"] = metrics["rejected_tokens"] / verified if verified else 0.0
    metrics["tokens_per_sec"] = verified / elapsed if elapsed > 0 else 0.0
    return metrics


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_rejection_mapping(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    tokenizer,
    gsm8k_examples: Sequence[Dict],
    *,
    draft_length: int,
    max_new_tokens: int,
    output_dir: Optional[Path],
    write_artifacts: bool,
    run_label: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    draft_device = first_parameter_device(draft_model)
    target_device = first_parameter_device(target_model)
    eos_token_id = tokenizer.eos_token_id

    metrics = build_metrics()
    rejection_logs: List[Dict] = []
    jsonl_handle = None
    jsonl_path = output_dir / JSONL_NAME if output_dir is not None else None

    if write_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when write_artifacts=True")
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")

    progress = tqdm(enumerate(gsm8k_examples), total=len(gsm8k_examples), desc=run_label)
    try:
        for prompt_idx, example in progress:
            prompt = example["question"]
            encoded = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)[0]
            input_length = int(encoded.numel())
            generated_ids = encoded.cpu()

            hidden_states_buffer: Dict[int, np.ndarray] = {}
            rejection_sequence: List[Dict] = []
            generated_token_ids: List[int] = []
            prompt_metrics = {"verified_tokens": 0, "accepted_tokens": 0, "rejected_tokens": 0}

            sync_devices([draft_device, target_device])
            start_time = time.perf_counter()

            with torch.inference_mode():
                step_index = 0
                stop_generation = False

                while int(generated_ids.numel()) < input_length + max_new_tokens:
                    base_len = int(generated_ids.numel())
                    base_new_tokens = base_len - input_length
                    draft_context = generated_ids.clone()
                    draft_candidates: List[int] = []
                    candidate_records: List[CandidateRecord] = []

                    for draft_step in range(draft_length):
                        if base_new_tokens + draft_step >= max_new_tokens:
                            break

                        draft_input = draft_context.unsqueeze(0).to(draft_device)
                        draft_out = draft_model(
                            input_ids=draft_input,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        final_hidden_state = draft_out.hidden_states[-1][0, -1, :]
                        draft_token_id = argmax_token(draft_out.logits[0, -1, :])
                        token_index = base_new_tokens + draft_step

                        candidate_records.append(
                            CandidateRecord(
                                prompt_id=prompt_idx,
                                step_index=step_index,
                                draft_step=draft_step,
                                token_index=token_index,
                                draft_token_id=draft_token_id,
                                hidden_state=final_hidden_state.detach()
                                .to(torch.float32)
                                .cpu()
                                .numpy(),
                            )
                        )
                        draft_candidates.append(draft_token_id)
                        draft_context = torch.cat(
                            [
                                draft_context,
                                torch.tensor([draft_token_id], dtype=torch.long),
                            ]
                        )

                        del draft_out, draft_input

                        if eos_token_id is not None and draft_token_id == eos_token_id:
                            break

                    if not draft_candidates:
                        break

                    target_context = torch.cat(
                        [
                            generated_ids,
                            torch.tensor(draft_candidates, dtype=torch.long),
                        ]
                    )
                    target_input = target_context.unsqueeze(0).to(target_device)
                    target_out = target_model(
                        input_ids=target_input,
                        output_hidden_states=False,
                        use_cache=False,
                    )
                    target_slice = target_out.logits[
                        0,
                        base_len - 1 : base_len - 1 + len(draft_candidates),
                        :,
                    ]
                    target_token_ids = torch.argmax(target_slice, dim=-1).detach().cpu().tolist()
                    del target_out, target_input

                    append_token_ids: List[int] = []
                    first_rejection_seen = False

                    for local_idx, record in enumerate(candidate_records):
                        draft_token_id = record.draft_token_id
                        target_token_id = int(target_token_ids[local_idx])
                        rejected = draft_token_id != target_token_id
                        accepted = not rejected

                        hidden_states_buffer[record.token_index] = record.hidden_state
                        log_entry = {
                            "prompt_id": record.prompt_id,
                            "step_index": record.step_index,
                            "draft_step": record.draft_step,
                            "token_index": record.token_index,
                            "prompt_length": input_length,
                            "context_length": base_len,
                            "draft_token_id": draft_token_id,
                            "target_token_id": target_token_id,
                            "draft_token": safe_decode(tokenizer, draft_token_id),
                            "target_token": safe_decode(tokenizer, target_token_id),
                            "accepted": accepted,
                            "rejected": rejected,
                            "hidden_state_shape": list(record.hidden_state.shape),
                            "hidden_state_archive": f"hidden_states_prompt_{prompt_idx:03d}.npz",
                            "hidden_state_key": f"token_{record.token_index}",
                        }

                        rejection_logs.append(log_entry)
                        rejection_sequence.append(
                            {
                                "token_index": record.token_index,
                                "draft_step": record.draft_step,
                                "draft_token_id": draft_token_id,
                                "target_token_id": target_token_id,
                                "accepted": accepted,
                                "rejected": rejected,
                            }
                        )

                        if jsonl_handle is not None:
                            jsonl_handle.write(json.dumps(log_entry) + "\n")

                        metrics["verified_tokens"] += 1
                        prompt_metrics["verified_tokens"] += 1

                        if accepted:
                            metrics["accepted_tokens"] += 1
                            prompt_metrics["accepted_tokens"] += 1
                            append_token_ids.append(draft_token_id)
                        else:
                            metrics["rejected_tokens"] += 1
                            prompt_metrics["rejected_tokens"] += 1
                            append_token_ids.append(target_token_id)
                            first_rejection_seen = True
                            break

                    if jsonl_handle is not None:
                        jsonl_handle.flush()

                    generated_ids = torch.cat(
                        [
                            generated_ids,
                            torch.tensor(append_token_ids, dtype=torch.long),
                        ]
                    )
                    generated_token_ids.extend(append_token_ids)

                    if any(token_id == eos_token_id for token_id in append_token_ids):
                        stop_generation = True
                    if int(generated_ids.numel()) >= input_length + max_new_tokens:
                        stop_generation = True

                    step_index += 1
                    if stop_generation:
                        break

                    if first_rejection_seen:
                        continue

            sync_devices([draft_device, target_device])
            elapsed = time.perf_counter() - start_time
            metrics["prompts"] += 1
            metrics["total_time_sec"] += elapsed

            if write_artifacts and output_dir is not None:
                hidden_states_npz = output_dir / f"hidden_states_prompt_{prompt_idx:03d}.npz"
                np.savez_compressed(
                    hidden_states_npz,
                    **{
                        f"token_{idx}": hidden_states_buffer[idx]
                        for idx in sorted(hidden_states_buffer)
                    },
                )

                prompt_summary = {
                    "prompt_id": prompt_idx,
                    "question": prompt,
                    "input_length": input_length,
                    "generated_token_ids": generated_token_ids,
                    "total_verified_tokens": prompt_metrics["verified_tokens"],
                    "accepted_tokens": prompt_metrics["accepted_tokens"],
                    "rejected_tokens": prompt_metrics["rejected_tokens"],
                    "rejection_sequence": rejection_sequence,
                    "elapsed_sec": elapsed,
                }
                write_json(output_dir / f"rejection_sequence_prompt_{prompt_idx:03d}.json", prompt_summary)

            current_alpha = (
                metrics["accepted_tokens"] / metrics["verified_tokens"]
                if metrics["verified_tokens"]
                else 0.0
            )
            progress.set_postfix(
                alpha=f"{current_alpha:.3f}",
                verified=int(metrics["verified_tokens"]),
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    return rejection_logs, finalize_metrics(metrics)


def load_rejection_logs(jsonl_path: Path) -> List[Dict]:
    logs: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def aggregate_rejection_patterns(logs: Sequence[Dict]) -> Dict:
    by_draft_step: Dict[int, Dict[str, int]] = {}
    by_token_index: Dict[int, Dict[str, int]] = {}

    for log in logs:
        rejected = bool(log["rejected"])
        for key_name, bucket in (
            ("draft_step", by_draft_step),
            ("token_index", by_token_index),
        ):
            key = int(log[key_name])
            bucket.setdefault(key, {"total": 0, "rejected": 0})
            bucket[key]["total"] += 1
            bucket[key]["rejected"] += int(rejected)

    draft_step_rates = {
        step: counts["rejected"] / counts["total"] if counts["total"] else 0.0
        for step, counts in sorted(by_draft_step.items())
    }
    token_index_rates = {
        idx: counts["rejected"] / counts["total"] if counts["total"] else 0.0
        for idx, counts in sorted(by_token_index.items())
    }

    rates = np.array(list(draft_step_rates.values()), dtype=np.float64)
    mean_rate = float(rates.mean()) if rates.size else 0.0
    std_rate = float(rates.std()) if rates.size else 0.0
    cv = std_rate / mean_rate if mean_rate > 0 else 0.0
    spread = float(rates.max() - rates.min()) if rates.size else 0.0

    if cv > 0.4 and spread > 0.15:
        assessment = "GREEN"
        reason = "PASS: CV > 0.4 and Spread > 0.15"
    elif cv < 0.15 or spread < 0.05:
        assessment = "RED"
        reason = "FAIL: CV < 0.15 or Spread < 0.05"
    else:
        assessment = "AMBIGUOUS"
        reason = "Between strict pass and fail gates"

    return {
        "draft_step_counts": by_draft_step,
        "token_index_counts": by_token_index,
        "draft_step_rejection_rates": draft_step_rates,
        "token_index_rejection_rates": token_index_rates,
        "mean_rejection_rate_by_draft_step": mean_rate,
        "std_rejection_rate_by_draft_step": std_rate,
        "cv": cv,
        "spread": spread,
        "assessment": assessment,
        "reason": reason,
    }


def plot_rejection_patterns(logs: Sequence[Dict], analysis: Dict, output_path: Path) -> None:
    draft_rates = analysis["draft_step_rejection_rates"]
    rejected_token_indices = [int(log["token_index"]) for log in logs if log["rejected"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    steps = list(draft_rates.keys())
    rates = [draft_rates[step] for step in steps]
    axes[0].bar([str(step) for step in steps], rates, color="#345995")
    axes[0].set_title("Rejection Rate by Draft Step")
    axes[0].set_xlabel("Draft step within K")
    axes[0].set_ylabel("Rejection rate")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)

    if rejected_token_indices:
        axes[1].hist(rejected_token_indices, bins=24, color="#d95d39", alpha=0.88)
        axes[1].set_ylabel("Frequency")
    else:
        axes[1].text(0.5, 0.5, "No rejected tokens", ha="center", va="center")
        axes[1].set_yticks([])
    axes[1].set_title("Rejected Token Index Histogram")
    axes[1].set_xlabel("Generated token index")
    axes[1].grid(axis="y", alpha=0.25)

    title = (
        f"Phase 0 Rejection Mapping: CV={analysis['cv']:.3f}, "
        f"Spread={analysis['spread']:.3f}, Gate={analysis['assessment']}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def collate_hidden_states_for_phase1a(
    rejection_logs: Sequence[Dict] | Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all logged hidden states from per-prompt .npz files and create (X, y).

    X has shape [num_verified_tokens, hidden_dim].
    y is 1 for rejected draft tokens and 0 for accepted draft tokens.
    """
    if isinstance(rejection_logs, Path):
        logs = load_rejection_logs(rejection_logs)
    else:
        logs = list(rejection_logs)

    by_prompt: Dict[int, List[Dict]] = {}
    for log in logs:
        by_prompt.setdefault(int(log["prompt_id"]), []).append(log)

    X: List[np.ndarray] = []
    y: List[int] = []

    for prompt_idx, prompt_logs in sorted(by_prompt.items()):
        npz_file = output_dir / f"hidden_states_prompt_{prompt_idx:03d}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing hidden-state archive: {npz_file}")

        with np.load(npz_file) as npz_data:
            for log in sorted(prompt_logs, key=lambda item: int(item["token_index"])):
                key = log.get("hidden_state_key") or f"token_{int(log['token_index'])}"
                if key not in npz_data:
                    raise KeyError(f"Missing {key} in {npz_file}")
                X.append(np.asarray(npz_data[key], dtype=np.float32))
                y.append(1 if log["rejected"] else 0)

    if not X:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    warmup_metrics: Dict[str, float],
    full_metrics: Optional[Dict[str, float]],
    analysis: Optional[Dict],
    collated_shape: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
) -> None:
    summary = {
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "tokenizer": args.tokenizer or args.target_model,
            "draft_device": args.draft_device,
            "target_device": args.target_device,
            "dtype": args.dtype,
            "draft_length": args.draft_length,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": args.num_examples,
            "warmup_examples": args.warmup_examples,
            "min_warmup_alpha": args.min_warmup_alpha,
            "seed": args.seed,
        },
        "warmup_metrics": warmup_metrics,
        "full_metrics": full_metrics,
        "gate_analysis": analysis,
        "collated_dataset_shape": {
            "X": list(collated_shape[0]),
            "y": list(collated_shape[1]),
        }
        if collated_shape
        else None,
    }
    write_json(output_dir / SUMMARY_NAME, summary)


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(f"{label}:")
    print(f"  prompts: {int(metrics['prompts'])}")
    print(f"  verified tokens: {int(metrics['verified_tokens'])}")
    print(f"  accepted tokens: {int(metrics['accepted_tokens'])}")
    print(f"  rejected tokens: {int(metrics['rejected_tokens'])}")
    print(f"  acceptance alpha: {metrics['acceptance_rate']:.4f}")
    print(f"  rejection rate: {metrics['rejection_rate']:.4f}")
    print(f"  throughput: {metrics['tokens_per_sec']:.2f} verified tokens/sec")
    print(f"  elapsed: {metrics['total_time_sec']:.2f} sec")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_dual_cuda(args.draft_device, args.target_device)
    print_gpu_resource_snapshot()
    prepare_output_dir(args.output_dir, args.overwrite)

    draft_model, target_model, tokenizer = load_models_and_tokenizer(args)
    examples = load_gsm8k_examples(args.num_examples)

    warmup_examples = examples[: args.warmup_examples]
    print(f"Running {len(warmup_examples)}-example acceptance warmup...")
    _, warmup_metrics = run_rejection_mapping(
        draft_model,
        target_model,
        tokenizer,
        warmup_examples,
        draft_length=args.draft_length,
        max_new_tokens=args.max_new_tokens,
        output_dir=None,
        write_artifacts=False,
        run_label="warmup",
    )
    print_metrics("Warmup metrics", warmup_metrics)

    if warmup_metrics["acceptance_rate"] < args.min_warmup_alpha:
        print(
            f"Stopping: warmup alpha {warmup_metrics['acceptance_rate']:.4f} "
            f"is below threshold {args.min_warmup_alpha:.4f}."
        )
        save_summary(args.output_dir, args, warmup_metrics, None, None)
        return 2

    print(f"Running full Phase 0 mapping on {len(examples)} examples...")
    rejection_logs, full_metrics = run_rejection_mapping(
        draft_model,
        target_model,
        tokenizer,
        examples,
        draft_length=args.draft_length,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        write_artifacts=True,
        run_label="phase0",
    )
    print_metrics("Full Phase 0 metrics", full_metrics)

    analysis = aggregate_rejection_patterns(rejection_logs)
    plot_path = args.output_dir / PLOT_NAME
    plot_rejection_patterns(rejection_logs, analysis, plot_path)

    collated_shape = None
    if not args.skip_collate:
        X, y = collate_hidden_states_for_phase1a(rejection_logs, args.output_dir)
        np.savez_compressed(args.output_dir / COLLATED_NAME, X=X, y=y)
        collated_shape = (X.shape, y.shape)
        print(f"Saved Phase 1a collator output: X{X.shape}, y{y.shape}")

    save_summary(args.output_dir, args, warmup_metrics, full_metrics, analysis, collated_shape)

    print("Gate analysis:")
    print(f"  CV: {analysis['cv']:.4f}")
    print(f"  Spread: {analysis['spread']:.4f}")
    print(f"  Assessment: {analysis['assessment']}")
    print(f"  Reason: {analysis['reason']}")
    print(f"Artifacts written to: {args.output_dir.resolve()}")
    print(f"Plot written to: {plot_path.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
