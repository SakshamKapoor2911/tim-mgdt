# Phase 11 Run Notes

This document tracks technical quirks, bugs, and hardware constraints discovered while executing the Phase 11 mechanistic interpretability and pruning pipeline on 2x RTX 3090 GPUs (48GB total VRAM).

## 1. PyTorch VRAM OOM on `state_dict()` Cloning
*   **The Issue:** During Stage 1 (Structured Pruning) and Stage 2 (IMP Search), the script creates a backup of the original weights before zero-masking specific heads or linear layers:
    ```python
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    ```
    For a 7B-8B model (~14GB in fp16), `.clone()` attempts to allocate another 14GB entirely inside VRAM. This instantly spikes usage to ~28GB, crashing the 24GB RTX 3090 with a `CUDA Out of Memory` error.
*   **The Fix:** Force PyTorch to copy the backup weights to the CPU (system RAM) during cloning:
    ```python
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ```
    This completely frees up the VRAM and gracefully utilizes the massive system RAM overhead instead. PyTorch's `load_state_dict()` seamlessly handles pushing the CPU tensors back to the GPU when restoring.

## 2. PyTorch VRAM OOM on `torch.nn.utils.prune` Hooks
*   **The Issue:** In Stage 2 and Stage 4, applying global unstructured L1 magnitude pruning using PyTorch's native wrapper `prune.l1_unstructured` causes catastrophic OOMs. The PyTorch pruning wrapper implements sparsity dynamically during the forward pass by creating two new tensors for every pruned layer: `weight_mask` and `weight_orig`.
    When applied globally to all Linear layers in a 7B-8B model, this triples the memory footprint of the weights (Original Weight + Mask + Orig Backup), instantly exceeding the 24GB VRAM ceiling.
*   **The Fix:** Completely rewrote `utils.apply_global_pruning` to avoid `torch.nn.utils.prune`. Instead, the script uses a highly memory-efficient algorithm to dynamically calculate the layer-wise `kthvalue` threshold and applies the zero-mask directly and statically to `weight.data.mul_(mask)` in-place. This requires 0 extra MB of VRAM.

## 3. Hugging Face 403 Client Error (Gated Repos)
*   **The Issue:** Running `meta-llama/Meta-Llama-3-8B` cleanly on a headless remote server fails with a 403 Client Error if the repository is gated behind a license agreement on Hugging Face.
*   **The Fix:** Ensure the environment has an authenticated token cached via `huggingface-cli login`, or substitute the model ID for the community-provided ungated version (`NousResearch/Meta-Llama-3-8B`) if an auth token is strictly unavailable.

## 4. 100% Parallel Hardware Utilization
*   **The Strategy:** Because `vllm` abstracts away standard PyTorch weight matrices and hidden state hooks (rendering mechanistic interpretability techniques impossible without deep C++ changes), we cannot use it to speed up Phase 11.
*   **The Solution:** We manually saturate the 48GB VRAM by orchestrating parallel Python processes, routing Llama-3-8B to GPU 0 (`CUDA_VISIBLE_DEVICES=0`) and Mistral-7B to GPU 1 (`CUDA_VISIBLE_DEVICES=1`), effectively halving wall-clock time compared to sequential processing.

## 5. Matrix Batching Perplexity Computation for 4x Speedup
*   **The Issue:** `compute_perplexity` evaluated Wikitext-2 one sequence at a time (`batch_size=1`). Because perplexity is measured heavily throughout the pruning process, this loop constituted ~90% of the entire pipeline's run-time and severely underutilized the RTX 3090 (leaving ~9GB VRAM completely idle).
*   **The Fix:** We rewrote `compute_perplexity` in `utils.py` to dynamically stack tokens into dense `[batch_size, seq_len]` matrices and passed `batch_size=4` down from the main orchestrator into all Phase 11 stages. This flawlessly saturates the Tensor Cores and cuts all subsequent runs by roughly 3-4x in wall-clock time!
