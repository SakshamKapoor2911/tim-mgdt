# COLM 2026 Actionable Interpretability — Submission Readiness Audit

> **Deadline: June 24, 2026 AOE** (5 days from now)
> **Format: Short paper (5 pages) or Long paper (9 pages), COLM LaTeX template, double-blind**

---

## 🟢 What's Already Strong (Workshop Fit)

The workshop explicitly asks for work that *moves interpretability from understanding to action*. Your paper's narrative does exactly this:

1. **Interpretability → Action pipeline is clear.** You measure layer sensitivity (interpretability), discover the Bathtub Curve (insight), and use it to allocate LoRA capacity (action). This is textbook "actionable interpretability."
2. **The Bathtub Curve is a genuine finding.** The ground-truth `delta_ce` values in [proxy_validation.json](file:///home/skapoor/tim-mgdt/experiments/phase12/proxy_validation.json) show layer 0 at δ=5.95, layer 1 at δ=2.86, layers 2-29 mostly at δ=0.03–0.09, and layers 30-31 spiking to δ=0.14 and δ=0.22. This is a real, dramatic signal.
3. **Surgical vs Uniform ablation is clean.** The headline result — Surgical LoRA beats Uniform LoRA on C4 (+12.2% vs +19.1%) with 5× fewer parameters — is a legitimate, well-controlled finding.
4. **Hardware accessibility story is compelling.** Consumer-GPU constraint is relatable and timely.
5. **Non-archival venue.** Workshop papers are non-archival, meaning this does not prevent you from submitting a stronger version to a main conference later.

---

## 🔴 Critical Weaknesses (Honest Assessment)

### 1. The Proxy Metrics Are Completely Broken — and the Paper Ignores This

This is the single biggest problem. Look at [proxy_validation.json](file:///home/skapoor/tim-mgdt/experiments/phase12/proxy_validation.json):

| Metric | Correlation with Ground Truth (ρ) | p-value |
|--------|-----------------------------------|---------|
| `ed` (Effective Dimensionality) | **NaN** | NaN |
| `jacobian` | **NaN** | NaN |
| `naudc` | **-0.142** | **0.439** |
| `ensemble` | **NaN** | NaN |

> [!CAUTION]
> **Every single proxy metric failed.** The `ed` and `jacobian` columns are constant 1.0 and 0.0 across all layers (bugs). The only surviving metric (`naudc`) has ρ = -0.14 with p = 0.44, meaning **it is statistically indistinguishable from random** and has the **wrong sign** (anti-correlated). 

**Why this matters for the paper:** You claim to use "proxy metrics" to discover the Bathtub Curve. But the proxies don't work. You actually used the **ground-truth `delta_ce`** (knocking out each layer and measuring real perplexity damage) to select layers. That is not a "proxy" — it's an exhaustive brute-force oracle measurement. A reviewer will immediately notice this disconnect.

**Fix (LOW effort):** Be honest. Reframe the paper: "We exhaustively measured per-layer pruning damage via knockout evaluation, discovered the Bathtub Curve, and used this ground-truth signal to allocate recovery capacity." Drop the proxy metric claims entirely. This is actually *fine* for a workshop paper — the finding and the action still stand.

---

### 2. SVD Initialization Actually Hurts Generalization

Your own data contradicts the abstract:

| Method | WT2 PPL | C4 PPL (Held-out) |
|--------|---------|-------------------|
| Surgical + Random Init | 5.635 | **8.975 (+12.3%)** |
| Surgical + SVD Init | 5.591 | **9.695 (+21.3%)** |

> [!WARNING]
> SVD initialization improves in-distribution WT2 by 0.8% but **worsens generalization by 9 percentage points**. On the metric that actually matters (held-out C4), SVD init is significantly worse than random init.

The paper currently acknowledges this in a small note but the abstract and title still center SVD-Initialization as the main contribution. A reviewer will see this as the method failing its core promise.

**Fix (LOW effort):** Reposition the contribution hierarchy:
1. **Primary contribution:** Surgical allocation guided by sensitivity analysis (this works cleanly)
2. **Secondary/negative result:** SVD initialization helps in-distribution but overfits, suggesting the pruning damage is not well approximated by a low-rank residual (interesting finding to discuss, not to sell as positive)

---

### 3. Training on WikiText-2 and Evaluating on WikiText-2 Test Is Contamination

Your LoRA training data is WikiText-2 train. Your "in-distribution" metric is WikiText-2 test. These share the same Wikipedia source distribution, and LoRA is memorizing distributional statistics. This is why every method "over-recovers" on WT2 (beating the dense baseline) while degrading on C4.

**Fix (ALREADY DONE):** You already identified this and added C4. But the paper draft still leads with WT2 numbers and treats WT2 as a primary metric. **C4 should be the primary metric in all tables.** WT2 should be secondary or in appendix.

---

### 4. Only 1 Model, 1 Architecture

You only test on Mistral-7B. Workshop papers can get away with this, but a reviewer may push back.

**Fix (MEDIUM effort, ~30 min):** Run the surgical-random-20pct ablation on Llama-3-8B. You already have the model downloaded and the infrastructure. One single run to show the Bathtub Curve generalizes across architectures would be very strong. You don't need the full sweep — just baseline + surgical at 20%.

---

### 5. Only 100 Training Steps (Convergence Not Proven)

100 steps is extremely short. The loss curve in [loss_curves.json](file:///home/skapoor/tim-mgdt/experiments/phase13/loss_curves.json) shows values bouncing between 1.0 and 2.2 with no clear convergence trend. A reviewer will ask: "How do you know 100 steps is enough? Would 500 steps change the Surgical vs Uniform gap?"

**Fix (LOW effort):** At minimum, run the Surgical-Random-20% config for 500 steps and show that the gap holds (or narrows). If it narrows, that's still an honest finding. If it holds, it massively strengthens the paper.

---

### 6. No Downstream Task Evaluation

Perplexity alone is an incomplete signal. Workshop reviewers in "Actionable Interpretability" will want to see whether the sensitivity-guided allocation actually recovers *task performance*, not just language modeling loss.

**Fix (MEDIUM effort):** Run a zero-shot accuracy check on 2-3 standard benchmarks (e.g., ARC-Easy, HellaSwag, WinoGrande) using `lm-eval-harness` for: (a) dense baseline, (b) pruned, (c) surgical-recovered. This is ~15 minutes of compute per eval and would massively strengthen the "actionable" claim.

---

## 📊 Priority Ranking: What To Do Before June 24

| Priority | Task | Effort | Impact | Status |
|----------|------|--------|--------|--------|
| **P0** | Reframe paper: drop proxy metric claims, lead with ground-truth sensitivity | 1 hour | 🔴 Critical (reviewer will reject on dishonesty) | Not done |
| **P0** | Reposition SVD init as secondary/negative result, not primary contribution | 1 hour | 🔴 Critical (data contradicts abstract) | Not done |
| **P0** | Make C4 the primary metric in all tables, demote WT2 | 30 min | 🔴 Critical (contamination framing) | Not done |
| **P1** | Run 500-step Surgical vs Uniform to prove convergence | ~20 min compute | 🟡 Important (reviewers will ask) | Not done |
| **P1** | Run Llama-3-8B surgical-random-20% for cross-architecture validation | ~15 min compute | 🟡 Important (single-model weakness) | Not done |
| **P2** | Add 2-3 downstream tasks via lm-eval-harness | ~1 hour compute | 🟡 Nice-to-have for "actionable" framing | Not done |
| **P2** | Visualize the Bathtub Curve as a figure (bar chart of δ_ce per layer) | 30 min | 🟢 Polish (reviewers love figures) | Not done |
| **P2** | Plot loss curves for Surgical vs Uniform showing convergence comparison | 30 min | 🟢 Polish | Not done |
| **P3** | Add EoRA as a baseline comparison (they open-sourced code) | 2-3 hours | 🟢 Would strengthen related work section | Not done |

---

## 🎯 Verdict: Is This Enough for COLM Actionable Interpretability?

**With P0 fixes: Yes, as a short paper (5 pages).** The core finding — that exhaustive layer sensitivity measurement reveals a Bathtub Curve, and surgically allocating LoRA to sensitive layers dramatically outperforms uniform allocation — is a clean, novel, and genuinely "actionable" result. That is your paper.

**Without P0 fixes: Likely reject.** If you submit with the current framing (claiming proxy metrics work, leading with SVD init that hurts generalization, WT2 as primary metric), a careful reviewer will find the data contradicts the claims.

**With P0 + P1 fixes: Competitive short paper.** Adding convergence proof and a second model makes this quite solid.

**With P0 + P1 + P2: Strong submission.** This would be a genuinely interesting workshop contribution.

---

## 💡 Reframed Title Suggestion

Instead of:
> "Surgical Recovery of Pruned LLMs: SVD-Initialized LoRA Guided by Layer Sensitivity"

Consider:
> **"The Bathtub Curve: How Layer Sensitivity Enables Surgical LoRA Recovery of Pruned LLMs on Consumer GPUs"**

This centers the interpretability insight (the Bathtub Curve) as the contribution, which is both honest and a better fit for the "Actionable Interpretability" workshop.
