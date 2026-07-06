# Phase 19 Termux Runbook

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

## Resume or verify the Phase 19 queue

```bash
ssh <user>@<gpu-host>
cd /home/skapoor/tim-mgdt
bash scripts/phase19_termux_control.sh ensure
bash scripts/phase19_termux_control.sh queue-phase20
```

`ensure` is idempotent. It leaves existing sessions alone, skips runs whose full
JSON result already exists, and starts only missing Phase 19 tmux jobs.
`queue-phase20` starts a waiting tmux session that does not consume GPUs until
the Phase 19 sessions have finished.

## Monitor from the phone

```bash
bash scripts/phase19_termux_control.sh status
bash scripts/phase19_termux_control.sh monitor 60
```

The main active sessions are:

```bash
tmux attach -t phase19_s30_full
tmux attach -t phase19_s40_full
tmux attach -t phase19_adaptive_boundary
```

Detach from tmux with `Ctrl-b`, then `d`.

## Tail logs

```bash
bash scripts/phase19_termux_control.sh logs s30
bash scripts/phase19_termux_control.sh logs s40
bash scripts/phase19_termux_control.sh logs adaptive
bash scripts/phase19_termux_control.sh logs phase20
```

## Current Phase 19 expectation

The intended queue is:

1. Full GSM8K eval for the 30% mask checkpoint on GPU 0.
2. Full GSM8K eval for the 40% mask checkpoint on GPU 1.
3. Adaptive boundary runner waits for both full evals.
4. Adaptive runner chooses 25% or 35%, runs the smoke check, then runs the full
   GSM8K eval and regenerates the Phase 19 dashboard/plot summaries.
5. Optional Phase 20 runner waits for all Phase 19 sessions, then tests whether
   the successful Phase 19 mask signal can guide real structured MLP-width
   pruning.

Run this before leaving the machine unattended:

```bash
bash scripts/phase19_termux_control.sh ensure
bash scripts/phase19_termux_control.sh queue-phase20
python scripts/phase19_status.py
```
