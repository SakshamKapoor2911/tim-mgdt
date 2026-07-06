# TIM-MGDT Termux Runbook

Termux should be used as the control terminal. The CUDA jobs must run on the
remote GPU host, inside tmux. Closing this chat, closing Termux, or turning off a
separate laptop will not stop those jobs, but powering off or suspending the GPU
host will.

## One-time Termux setup

```bash
pkg update
pkg install openssh tmux
```

If you use SSH keys from Termux:

```bash
ssh-keygen -t ed25519
ssh-copy-id <user>@<gpu-host>
```

## Current one-command status

Use this first:

```bash
ssh <user>@<gpu-host>
cd /home/skapoor/tim-mgdt
python scripts/phase19_status.py
```

This prints tmux sessions, GPU use, active run progress, Phase 19/20/21/22/23
tails, and the current compact result tables. For continuous monitoring:

```bash
bash scripts/phase19_termux_control.sh monitor 60
```

## Resume or verify queued runs

```bash
ssh <user>@<gpu-host>
cd /home/skapoor/tim-mgdt
bash scripts/phase19_termux_control.sh ensure
bash scripts/phase19_termux_control.sh queue-phase20
bash scripts/phase19_termux_control.sh queue-phase23
bash scripts/phase19_termux_control.sh queue-publish
```

`ensure` is idempotent. It leaves existing sessions alone, skips runs whose full
JSON result already exists, and starts only missing Phase 19 tmux jobs.
`queue-phase20` starts a waiting tmux session that does not consume GPUs until
the Phase 19 sessions have finished.
`queue-phase23` starts the global unstructured magnitude baseline if it is not
already running and the full GSM8K result is missing.
`queue-publish` waits for Phase 19 and Phase 20 sessions, rebuilds compact
summaries/plots/dashboard artifacts, commits them, and pushes to GitHub. It does
not stage checkpoints, logs, or sample JSONL files.

The main active sessions are:

```bash
tmux attach -t phase23_global_s20
```

Detach from tmux with `Ctrl-b`, then `d`.

## Tail logs

```bash
bash scripts/phase19_termux_control.sh logs s30
bash scripts/phase19_termux_control.sh logs s40
bash scripts/phase19_termux_control.sh logs adaptive
bash scripts/phase19_termux_control.sh logs phase20
bash scripts/phase19_termux_control.sh logs phase21
bash scripts/phase19_termux_control.sh logs phase22
bash scripts/phase19_termux_control.sh logs phase23
bash scripts/phase19_termux_control.sh logs publish
```

## Current Stage: Phase 23 Complete; Paper Packaging

Phase 23 answered the obvious reviewer question, "is the 20% result just
unstructured magnitude pruning rather than a layerwise dense-mask diagnostic?"

Completed result:

```bash
global 20% mask: 35.33% strict EM / 35.63% flexible EM on full GSM8K
delta vs dense: -3.34 pt
delta vs layerwise 20% mask: -1.90 pt
delta vs structural 20% width pruning: +30.10 pt
C4 PPL: 8.203
actual linear sparsity: 20.06%
```

Artifact paths:

```bash
metadata: experiments/phase23_global_unstructured/Mistral-7B-v0.1_mask_pruning_global_s20.json
eval result: experiments/phase23_global_unstructured/gsm8k_eval_global_s20_full_2026-07-06T16-36-08_2026-07-06T14-02-18.501393.json
```

Interpretation: global 20% unstructured magnitude masking does not collapse; it
retains substantial GSM8K reasoning and is far stronger than structural width
pruning. It is still weaker than the layerwise 20% dense-shaped mask, so
layerwise sparsity allocation matters within geometry-preserving unstructured
masking.

## Next Steps

1. Keep `paper/main.tex` as the authoritative submission draft.
2. Rebuild the Overleaf zip at `paper/tim-mgdt-overleaf-paper.zip`.
3. Run:

```bash
python -m py_compile code/phases/phase11/utils.py code/phases/phase19/mask_pruning.py scripts/phase19_status.py
pytest tests/test_phase11_real.py tests/test_phase20_mask_guided.py
python scripts/phase19_readiness.py
```

4. Commit and push compact code, paper, runbook, dashboard, and Phase 23 JSON
   artifacts. Do not stage checkpoint directories, run logs, or sample JSONL
   files.
5. Optional next experiment: per-layer or per-module ablation around the 20--25%
   boundary to identify where layerwise allocation matters.

Run this before leaving the machine unattended:

```bash
bash scripts/phase19_termux_control.sh ensure
bash scripts/phase19_termux_control.sh queue-phase20
bash scripts/phase19_termux_control.sh queue-phase23
bash scripts/phase19_termux_control.sh queue-publish
python scripts/phase19_status.py
```
