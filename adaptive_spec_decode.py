#!/usr/bin/env python3
"""
Adaptive speculative decoding experiment for the Ghost Subnetwork pipeline.

This evaluates whether a Phase 1a router trained on draft hidden states can
choose shorter or longer speculative spans while preserving exact target-greedy
outputs. The target model still verifies every proposed span, so correctness is
checked by comparing generated token ids to target-only greedy decoding.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B"
DEFAULT_ROUTER_DIR = Path("rejection_mapping_logs_20260623_094351")
DEFAULT_OUTPUT_DIR = Path("adaptive_spec_results")


@dataclass
class DraftCandidate:
    token_id: int
    token_index: int
    reject_probability: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed and router-adaptive speculative decoding.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--router-dir", type=Path, default=DEFAULT_ROUTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draft-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. eager is slower but avoids backend-dependent exactness drift.",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--static-k", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--adaptive-max-k", type=int, default=8)
    parser.add_argument("--adaptive-thresholds", type=float, nargs="+", default=[0.50, 0.70, 0.8625])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
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
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This experiment expects two CUDA GPUs.")
    if torch.device(draft_device).index == torch.device(target_device).index:
        raise RuntimeError("Draft and target devices must be different GPUs.")


def sync_devices(devices: Iterable[torch.device]) -> None:
    seen = set()
    for device in devices:
        if device.type != "cuda":
            continue
        idx = device.index
        if idx in seen:
            continue
        torch.cuda.synchronize(device)
        seen.add(idx)


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
    attn_implementation: str,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        attn_implementation=attn_implementation,
    )
    model.eval()
    model.config.use_cache = False
    return model


def load_examples(num_prompts: int) -> List[Dict]:
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_prompts}]")
    return [{"question": row["question"], "answer": row.get("answer", "")} for row in dataset]


def model_inputs_from_cpu(ids: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    input_ids = ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def argmax_token(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def router_probability(router, hidden_state: torch.Tensor) -> float:
    feature = hidden_state.detach().to(torch.float32).cpu().numpy().reshape(1, -1)
    return float(router.predict_proba(feature)[0, 1])


def target_only_generate(
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int],
) -> Tuple[List[int], Dict]:
    target_device = first_parameter_device(target_model)
    generated_ids = input_ids.cpu()
    generated_tokens: List[int] = []
    metrics = {
        "target_forward_calls": 0,
        "draft_forward_calls": 0,
        "verified_candidates": 0,
        "accepted_candidates": 0,
        "rejected_candidates": 0,
        "drafted_candidates": 0,
        "verification_batches": 0,
    }

    with torch.inference_mode():
        while len(generated_tokens) < max_new_tokens:
            target_inputs = model_inputs_from_cpu(generated_ids, target_device)
            target_out = target_model(**target_inputs, use_cache=False)
            token_id = argmax_token(target_out.logits[0, -1, :])
            del target_out, target_inputs

            metrics["target_forward_calls"] += 1
            generated_tokens.append(token_id)
            generated_ids = torch.cat([generated_ids, torch.tensor([token_id], dtype=torch.long)])
            if eos_token_id is not None and token_id == eos_token_id:
                break

    return generated_tokens, metrics


def draft_span(
    draft_model: torch.nn.Module,
    router,
    generated_ids: torch.Tensor,
    *,
    max_span: int,
    max_remaining: int,
    mode: str,
    threshold: Optional[float],
) -> Tuple[List[DraftCandidate], Dict]:
    draft_device = first_parameter_device(draft_model)
    draft_context = generated_ids.cpu().clone()
    candidates: List[DraftCandidate] = []
    metrics = {"draft_forward_calls": 0, "router_calls": 0, "router_stops": 0}

    with torch.inference_mode():
        for draft_step in range(min(max_span, max_remaining)):
            draft_inputs = model_inputs_from_cpu(draft_context, draft_device)
            draft_out = draft_model(
                **draft_inputs,
                output_hidden_states=(mode == "adaptive"),
                use_cache=False,
            )
            token_id = argmax_token(draft_out.logits[0, -1, :])
            reject_prob = None

            if mode == "adaptive":
                hidden_state = draft_out.hidden_states[-1][0, -1, :]
                reject_prob = router_probability(router, hidden_state)
                metrics["router_calls"] += 1

            del draft_out, draft_inputs

            candidates.append(
                DraftCandidate(
                    token_id=token_id,
                    token_index=int(generated_ids.numel()) + draft_step,
                    reject_probability=reject_prob,
                )
            )
            draft_context = torch.cat([draft_context, torch.tensor([token_id], dtype=torch.long)])
            metrics["draft_forward_calls"] += 1

            if mode == "adaptive" and threshold is not None and reject_prob is not None:
                if reject_prob >= threshold:
                    metrics["router_stops"] += 1
                    break

    return candidates, metrics


def speculative_generate(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    router,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    max_span: int,
    mode: str,
    threshold: Optional[float] = None,
) -> Tuple[List[int], Dict]:
    target_device = first_parameter_device(target_model)
    generated_ids = input_ids.cpu()
    generated_tokens: List[int] = []
    metrics = {
        "target_forward_calls": 0,
        "draft_forward_calls": 0,
        "router_calls": 0,
        "router_stops": 0,
        "verified_candidates": 0,
        "accepted_candidates": 0,
        "rejected_candidates": 0,
        "drafted_candidates": 0,
        "verification_batches": 0,
        "accepted_span_lengths": [],
        "proposed_span_lengths": [],
        "router_probabilities": [],
    }

    with torch.inference_mode():
        while len(generated_tokens) < max_new_tokens:
            max_remaining = max_new_tokens - len(generated_tokens)
            candidates, draft_metrics = draft_span(
                draft_model,
                router,
                generated_ids,
                max_span=max_span,
                max_remaining=max_remaining,
                mode=mode,
                threshold=threshold,
            )

            if not candidates:
                break

            for key, value in draft_metrics.items():
                metrics[key] += value

            draft_token_ids = [candidate.token_id for candidate in candidates]
            metrics["drafted_candidates"] += len(draft_token_ids)
            metrics["proposed_span_lengths"].append(len(draft_token_ids))
            metrics["router_probabilities"].extend(
                [
                    candidate.reject_probability
                    for candidate in candidates
                    if candidate.reject_probability is not None
                ]
            )

            # We verify K candidates without using the speculative "bonus" token.
            # Therefore the target only needs prefix + the first K-1 candidates:
            # logits at positions [n-1, ..., n+K-2] predict candidates [1, ..., K].
            # Including the token being predicted can introduce implementation-level
            # drift under some attention kernels, even though it is causally masked.
            if len(draft_token_ids) > 1:
                target_context = torch.cat(
                    [
                        generated_ids,
                        torch.tensor(draft_token_ids[:-1], dtype=torch.long),
                    ]
                )
            else:
                target_context = generated_ids
            base_len = int(generated_ids.numel())
            target_inputs = model_inputs_from_cpu(target_context, target_device)
            target_out = target_model(**target_inputs, use_cache=False)
            target_slice = target_out.logits[0, base_len - 1 : base_len - 1 + len(draft_token_ids), :]
            target_token_ids = torch.argmax(target_slice, dim=-1).detach().cpu().tolist()
            del target_out, target_inputs

            metrics["target_forward_calls"] += 1
            metrics["verification_batches"] += 1

            append_token_ids: List[int] = []
            accepted_this_batch = 0
            for draft_token_id, target_token_id in zip(draft_token_ids, target_token_ids):
                metrics["verified_candidates"] += 1
                if draft_token_id == int(target_token_id):
                    append_token_ids.append(draft_token_id)
                    accepted_this_batch += 1
                    metrics["accepted_candidates"] += 1
                else:
                    append_token_ids.append(int(target_token_id))
                    metrics["rejected_candidates"] += 1
                    break

            metrics["accepted_span_lengths"].append(accepted_this_batch)
            generated_ids = torch.cat(
                [generated_ids, torch.tensor(append_token_ids, dtype=torch.long)]
            )
            generated_tokens.extend(append_token_ids)

            if eos_token_id is not None and any(token_id == eos_token_id for token_id in append_token_ids):
                break

    return generated_tokens[:max_new_tokens], metrics


def finalize_run_metrics(raw_metrics: Dict, elapsed_sec: float, generated_tokens: int) -> Dict:
    verified = raw_metrics["verified_candidates"]
    accepted = raw_metrics["accepted_candidates"]
    target_calls = raw_metrics["target_forward_calls"]
    draft_calls = raw_metrics["draft_forward_calls"]
    proposed_spans = raw_metrics.get("proposed_span_lengths", [])
    accepted_spans = raw_metrics.get("accepted_span_lengths", [])
    router_probs = raw_metrics.get("router_probabilities", [])

    metrics = dict(raw_metrics)
    metrics["elapsed_sec"] = elapsed_sec
    metrics["generated_tokens"] = generated_tokens
    metrics["tokens_per_sec"] = generated_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
    metrics["acceptance_rate"] = accepted / verified if verified else 0.0
    metrics["target_forwards_per_generated_token"] = (
        target_calls / generated_tokens if generated_tokens else 0.0
    )
    metrics["draft_forwards_per_generated_token"] = (
        draft_calls / generated_tokens if generated_tokens else 0.0
    )
    metrics["mean_proposed_span"] = float(np.mean(proposed_spans)) if proposed_spans else 0.0
    metrics["mean_accepted_span"] = float(np.mean(accepted_spans)) if accepted_spans else 0.0
    metrics["mean_router_probability"] = float(np.mean(router_probs)) if router_probs else None
    metrics["p95_router_probability"] = float(np.percentile(router_probs, 95)) if router_probs else None
    return metrics


def sum_metric_dicts(metric_dicts: Sequence[Dict]) -> Dict:
    totals: Dict = {}
    list_keys = {"accepted_span_lengths", "proposed_span_lengths", "router_probabilities"}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if key in list_keys:
                totals.setdefault(key, []).extend(value)
            elif isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return totals


def run_method(
    method_name: str,
    examples: Sequence[Dict],
    tokenizer,
    draft_model: Optional[torch.nn.Module],
    target_model: torch.nn.Module,
    router,
    *,
    max_new_tokens: int,
    mode: str,
    max_span: int = 1,
    threshold: Optional[float] = None,
    reference_outputs: Optional[Sequence[List[int]]] = None,
) -> Dict:
    draft_device = first_parameter_device(draft_model) if draft_model is not None else None
    target_device = first_parameter_device(target_model)
    eos_token_id = tokenizer.eos_token_id
    prompt_metrics: List[Dict] = []
    outputs: List[List[int]] = []
    exact_matches: List[bool] = []

    sync_targets = [target_device] + ([draft_device] if draft_device is not None else [])
    sync_devices(sync_targets)
    start_time = time.perf_counter()

    for idx, example in tqdm(list(enumerate(examples)), desc=method_name):
        input_ids = tokenizer.encode(
            example["question"],
            return_tensors="pt",
            add_special_tokens=True,
        )[0]

        if mode == "target":
            output_ids, metrics = target_only_generate(
                target_model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        else:
            if draft_model is None:
                raise ValueError("draft_model is required for speculative methods")
            output_ids, metrics = speculative_generate(
                draft_model,
                target_model,
                router,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                max_span=max_span,
                mode=mode,
                threshold=threshold,
            )

        outputs.append(output_ids)
        prompt_metrics.append(metrics)
        if reference_outputs is not None:
            exact_matches.append(output_ids == reference_outputs[idx])

    sync_devices(sync_targets)
    elapsed_sec = time.perf_counter() - start_time
    raw_totals = sum_metric_dicts(prompt_metrics)
    generated_tokens = sum(len(output) for output in outputs)
    metrics = finalize_run_metrics(raw_totals, elapsed_sec, generated_tokens)
    metrics["method"] = method_name
    metrics["prompts"] = len(examples)
    metrics["exact_match_rate"] = (
        sum(exact_matches) / len(exact_matches) if exact_matches else None
    )
    metrics["mismatched_prompt_ids"] = (
        [idx for idx, matched in enumerate(exact_matches) if not matched]
        if exact_matches
        else []
    )
    return {"metrics": metrics, "outputs": outputs}


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    ensure_dual_cuda(args.draft_device, args.target_device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(args.dtype)
    tokenizer_name = args.tokenizer or args.target_model

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = load_tokenizer(tokenizer_name, args.local_files_only)
    print(f"Loading router from: {args.router_dir}")
    router = joblib.load(args.router_dir / "phase1a_router_model.joblib")

    print(f"Loading draft on {args.draft_device}: {args.draft_model}")
    draft_model = load_model_on_device(
        args.draft_model,
        args.draft_device,
        dtype,
        args.local_files_only,
        args.attn_implementation,
    )
    print(f"Loading target on {args.target_device}: {args.target_model}")
    target_model = load_model_on_device(
        args.target_model,
        args.target_device,
        dtype,
        args.local_files_only,
        args.attn_implementation,
    )

    examples = load_examples(args.num_prompts)
    print(
        f"Running adaptive speculation experiment on {len(examples)} prompts, "
        f"max_new_tokens={args.max_new_tokens}"
    )

    results: Dict[str, Dict] = {
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "router_dir": str(args.router_dir),
            "draft_device": args.draft_device,
            "target_device": args.target_device,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "static_k": args.static_k,
            "adaptive_max_k": args.adaptive_max_k,
            "adaptive_thresholds": args.adaptive_thresholds,
            "seed": args.seed,
        },
        "methods": {},
    }

    target_run = run_method(
        "target_greedy",
        examples,
        tokenizer,
        None,
        target_model,
        router,
        max_new_tokens=args.max_new_tokens,
        mode="target",
    )
    reference_outputs = target_run["outputs"]
    results["methods"]["target_greedy"] = target_run["metrics"]
    write_json(args.output_dir / "adaptive_spec_results_partial.json", results)

    for k in args.static_k:
        method_name = f"static_k{k}"
        run = run_method(
            method_name,
            examples,
            tokenizer,
            draft_model,
            target_model,
            router,
            max_new_tokens=args.max_new_tokens,
            mode="static",
            max_span=k,
            reference_outputs=reference_outputs,
        )
        results["methods"][method_name] = run["metrics"]
        write_json(args.output_dir / "adaptive_spec_results_partial.json", results)

    for threshold in args.adaptive_thresholds:
        method_name = f"adaptive_t{threshold:.4f}".replace(".", "p")
        run = run_method(
            method_name,
            examples,
            tokenizer,
            draft_model,
            target_model,
            router,
            max_new_tokens=args.max_new_tokens,
            mode="adaptive",
            max_span=args.adaptive_max_k,
            threshold=threshold,
            reference_outputs=reference_outputs,
        )
        results["methods"][method_name] = run["metrics"]
        write_json(args.output_dir / "adaptive_spec_results_partial.json", results)

    target_tps = results["methods"]["target_greedy"]["tokens_per_sec"]
    target_forwards_per_token = results["methods"]["target_greedy"][
        "target_forwards_per_generated_token"
    ]
    for metrics in results["methods"].values():
        metrics["wall_clock_speedup_vs_target"] = (
            metrics["tokens_per_sec"] / target_tps if target_tps > 0 else 0.0
        )
        metrics["target_forward_reduction_vs_target"] = (
            1.0
            - (
                metrics["target_forwards_per_generated_token"]
                / target_forwards_per_token
            )
            if target_forwards_per_token > 0
            else 0.0
        )

    write_json(args.output_dir / "adaptive_spec_results.json", results)

    print("Adaptive speculation experiment complete")
    for name, metrics in results["methods"].items():
        print(
            f"  {name}: tps={metrics['tokens_per_sec']:.2f}, "
            f"speedup={metrics['wall_clock_speedup_vs_target']:.3f}x, "
            f"target_fw/token={metrics['target_forwards_per_generated_token']:.3f}, "
            f"target_fw_reduction={metrics['target_forward_reduction_vs_target']:.3f}, "
            f"match={metrics['exact_match_rate']}"
        )
    print(f"Wrote: {args.output_dir / 'adaptive_spec_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
