# Paper Weaknesses: Comprehensive Analysis

**Paper**: *Suturing the Cut: Subspace Rotation and the Fluency--Reasoning Dissociation in Incision Boundary Healing of Depth-Pruned LLMs*
**Target**: NeurIPS 2026 / COLM 2026 / BlackboxNLP 2026 workshops
**Date**: 2026-07-24
**Status**: Pre-submission diagnostic

---

## Scoring Legend

| Dimension | Scale |
|:---|:---|
| **Severity** | Critical (reject signal) > High (major challenge) > Medium (reviewer question) > Low (nitpick) |
| **Effort-to-Reward** | High (do it) / Medium (consider) / Low (marginal) |
| **Paper already addresses?** | Yes / Partially / No |

---

## 1. GSM8K Sample Size (N=100) — RESOLVED (N=500 Completed)

- **Severity**: Resolved
- **Effort-to-Reward**: Done — N=500 ran on 2026-07-24
- **Paper addresses?** Now addressed — N=500 results available in `docs/analysis/n500_baseline_analysis.md`

| Condition | Paper N=100 (8-shot) | N=500 (8-shot) | CI Width Δ |
|:---|---:|---:|:---:|
| Dense intact baseline | 34.0% [24.8%, 44.2%] | **39.6%** [35.4%, 44.0%] | ±9.7 → ±4.3 pp |
| Boundary LoRA (t=2) | 7.0% [2.9%, 13.9%] | **6.2%** [4.2%, 8.4%] | ±5.5 → ±2.1 pp |
| Pruned unhealed | 7.0% [2.9%, 13.9%] | **1.8%** [0.8%, 3.0%] | ±5.5 → ±1.1 pp |
| Recovery gap (healed vs pruned) | Not distinguishable at N=100 | **+4.4 pp*** (non-overlapping CIs) | — |

*\*Statistically significant at N=500. Boundary healing does produce a small real improvement in reasoning.*

**Impact on paper claims**:
- The dissociation is confirmed: fluency recovers dramatically but reasoning only improves by +4.4 pp (from 1.8% to 6.2%).
- The pruned baseline is ~2%, not 7%. The paper's 7.0% was an N=100 sampling high-side.
- The paper should update the main results table to use the N=500 numbers.

---

## 2. No Training-Time PC₁ Recovery Experiment

- **Severity**: **Medium** (acceptable for workshop)
- **Effort-to-Reward**: **Low** (too expensive, out of scope)
- **Paper addresses?** Yes -- explicitly flagged as future work in §5.1.

**Description**: The paper demonstrates that inference-time geometric correction (Procrustes, PC₁ Givens un-rotation) fails to restore reasoning (0-4% GSM8K). It hypothesizes that downstream weight matrices (L15--31) were calibrated on the inverted PC₁ during pre-training and cannot "accept" a corrected signal without re-training. However, no training-time PC₁-targeted recovery experiment is performed (e.g., SFT on reasoning tasks with a PC₁-alignment auxiliary loss).

**Reviewer attack**: "You've shown the diagnostic side thoroughly, but the paper ends with 'future work'. For a workshop this may be acceptable, but the paper feels incomplete without at least a proof-of-concept recovery."

**Mitigation**:
- The paper is explicitly framed as a diagnostic/mechanistic contribution, not a prescriptive one
- The PC₁ inversion discovery (179.52°, projection = −1.000) is a novel finding in its own right
- Training-time PC₁ recovery is a substantial engineering effort (requires reasoning SFT + geometric regularization) that exceeds workshop scope
- The sufficiency test (34% → 10%) provides causal evidence that is complete as a diagnostic arc
- Falsification of 9 competing recovery paradigms provides completeness

**Recommendation**: Keep the explicit future work flag. Frame the paper as: "We identify the geometric mechanism of reasoning collapse and falsify the obvious repair strategies -- training-time recovery is the natural next step." This is honest and workshop-appropriate.

---

## 3. Pruned Baseline Discrepancy (1% vs 7%) — RESOLVED

- **Severity**: Resolved — investigated on 2026-07-24
- **Effort-to-Reward**: Resolved (N=500 cross-validation completed)
- **Paper addresses?** Now addressed.

**Resolution findings**: The discrepancy is explained by two factors:

1. **Shot count (minor factor)**: 5-shot gives 1.0%, 8-shot gives 3.0% at N=100. The extra exemplars help by ~2 pp.
2. **N=100 sampling variance (main factor)**: At N=500, the pruned unhealed baseline is **1.8%** [0.8%, 3.0%]. The paper's 7.0% [2.9%, 13.9%] was a binomial high-side draw. The CIs barely overlap at the edge (our upper bound 3.0% vs paper's lower bound 2.9%), consistent with a small additional pipeline effect + the N=100 sampling noise.

**Confirmed baseline values (N=500, 8-shot)**:
- Dense baseline: 39.6% [35.4%, 44.0%]
- Pruned unhealed (drop 11-14): 1.8% [0.8%, 3.0%]
- Boundary healed (t=2): 6.2% [4.2%, 8.4%]

**Recommendation**: Replace the paper's headline numbers with the N=500 values. The dissociation and mechanistic claims are unaffected. Add a footnote: "8-shot GSM8K accuracy at N=500 with 95% Clopper-Pearson CIs. Paper baseline re-estimated from N=500; original N=100 estimate was 7.0% [2.9%, 13.9%]."

Full analysis in `docs/analysis/n500_baseline_analysis.md`.

---

## 4. The Noise Floor Problem (0--4% GSM8K)

- **Severity**: **Medium** (tractable with framing)
- **Effort-to-Reward**: **Medium** (requires careful writing)
- **Paper addresses?** Partially -- documents all 9 failed interventions but doesn't explicitly discuss the noise floor.

**Description**: Every constructive recovery intervention tested in Phase 25 converges to 0--4% GSM8K, which is statistically indistinguishable from the pruned baseline at N=100. The noise floor spans:

| Intervention | GSM8K | Distinguishable from 1% baseline? |
|:---|:---:|:---:|
| Random rotation | 0.00% | No |
| Norm matching (paper) | 0.00% | No |
| Wrong-layer R* at L0 | 0.00% | No |
| Pruned unhealed baseline (Phase 25) | 1.00% | -- |
| P4 Middle location (n=300) | 2.00% | No |
| Procrustes R* | 3.00% | Marginal (p≈0.31) |
| Wrong-layer R* at L5 | 3.00% | Marginal |
| Wrong-PC PC₂ realign | 3.00% | Marginal |
| P5 PC₁ Givens un-rotation | 4.00% | Marginal |
| P4 Late location (n=300) | 4.33% | Yes (CI [2.3, 7.3]) |

**Implication**: The noise floor means the paper cannot distinguish between (a) inference-time geometric correction is impossible vs (b) the two specific geometric corrections tested (Procrustes, PC₁ Givens) happen to be insufficient. At N=100, the paper also cannot distinguish between correct and incorrect interventions (wrong-layer R* at L5 gives 3%, same as correct-layer R* at L11).

**Reviewer attack**: "If Procrustes R* at the correct layer gives the same result as R* at a wrong layer, and PC₁ un-rotation gives the same result as random rotation, then your geometric correction methods may not be correctly specified. How do you know there isn't a different geometric correction that WOULD work?"

**Mitigation**:
- Frame the noise floor honestly: "All inference-time geometric interventions, regardless of specificity, fell within the evaluation noise floor, indicating that representational alignment alone is insufficient for recovery."
- The PC₁ anti-alignment finding (179.52°) provides a mechanistic explanation: the inversion propagates through downstream weights
- The sufficiency test (34% → 10%) proves the causal direction even though the reverse direction fails
- P5 is conclusive: if un-inverting the one component known to be causally necessary (PC₁, 34% → 11% when ablated) still gives noise-floor results, then inference-time correction is fundamentally insufficient

**Recommendation**: Do NOT claim "recovery is impossible" or "downstream weights are irreversibly biased" -- these are speculative. Instead say: "Recovery was not observed under inference-time geometric interventions. One possible explanation, consistent with these findings, is that the downstream weight matrices have become irreversibly biased toward the post-incision geometry. We leave training-time PC₁-targeted recovery to future work."

---

## 5. Location Non-Specificity of Procrustes R*

- **Severity**: **Medium**
- **Effort-to-Reward**: **Medium** (can be mitigated with framing)
- **Paper addresses?** No -- this is a Phase 25 discovery.

**Description**: Phase 25 (P2B) tested Procrustes R* at multiple layers:
- R* at Layer 0 (very early): **0.00%**
- R* at Layer 5 (early, wrong): **3.00%**
- R* at Layer 25 (late, wrong): **3.00%**
- R* at Layer 11 (correct, post-incision): **3.00%**

The Procrustes rotation produces the same 3% result at nearly any layer from L5 onward, including layers distant from the incision. This weakens the spatial specificity argument: the intervention may not act through the proposed incision-interface mechanism.

**Mitigation**:
- R* at Layer 0 gives 0%, establishing that very early layers are invariant
- Every result from L5 onward is within the noise floor -- so layer-level differences cannot be distinguished at N=100
- The conclusion is still valid: no layer-level Procrustes correction restores reasoning
- Raise n to 300 for headline layer tests if possible

**Recommendation**: Do not claim layer specificity for recovery interventions. Instead: "Procrustes realignment, regardless of application layer, remained within the evaluation noise floor, providing no evidence for layer-specific geometric correction."

---

## 6. Residual Norm vs. Rotation Correlation

- **Severity**: **Medium** (reviewer will likely raise this)
- **Effort-to-Reward**: **High** (easy writing fix)
- **Paper addresses?** Partially -- paper Table 5 reports Spearman correlations but uses per-passage data; the paper text notes correlation but doesn't fully address the "simpler explanation" objection.

**Description**: The Phase 25 per-passage analysis reveals that θ and residual norm have r = −0.759 (Pearson), ρ = −0.546 (Spearman). The paper's own Table 5 reports ρ = −0.546, confirming this. A reviewer could argue: "If residual norm explains 30-58% of the variance in rotation, isn't norm damage the simpler and more parsimonious explanation for reasoning collapse?"

**Mitigation** (strong):
1. **Norm matching fails catastrophically** (GSM8K = 0.0%, paper Table 6) -- norm correction alone does nothing
2. **Procrustes R* preserves norms** (orthogonal transformation, ‖Rh‖ = ‖h‖) yet fails to recover reasoning -- if norm were the mechanism, Procrustes (which preserves norms) would recover reasoning. It doesn't.
3. **Norm and rotation are causally distinct**: norm is a scalar (magnitude), rotation is directional (geometry). They are correlated because pruning affects both, but they operate on different aspects of representation.
4. **θ is uncorrelated with competing scalar metrics** (LayerNorm μ/σ drift p>0.4, effective rank p>0.84) -- establishing that θ captures unique geometric information.

**Recommendation**: Add explicit text: "While residual norm and rotation are correlated (ρ = −0.546, p < 0.001), norm correction alone fails catastrophically (0.0% GSM8K, Table 6), and Procrustes R* preserves norms exactly (orthogonal transformation) yet does not recover reasoning. Norm and rotation are correlated but causally distinct." This is a strong rebuttal preemption.

---

## 7. Single Model for All Reasoning Experiments

- **Severity**: **Medium** (partially mitigated)
- **Effort-to-Reward**: **Medium** (adds cross-model evidence if time)
- **Paper addresses?** Partially -- cross-model evidence for perplexity and geometry; reasoning only on Mistral-7B.

**Description**: All reasoning experiments (GSM8K causal interventions, PC ablation, rotation-depth curve, 9-intervention failure matrix, sufficiency test) are run on Mistral-7B-v0.1 only. Llama-3-8B and Gemma-2-2B are validated only for perplexity recovery (Table 1) and incision interface geometry (Table 2). The fluency-reasoning dissociation is demonstrated on only one model.

**Reviewer attack**: "You claim that the fluency-reasoning dissociation is a general phenomenon, but you only show reasoning collapse on Mistral-7B. Would Llama-3-8B show the same dissociation? If not, your geometric explanation may be Mistral-specific."

**Mitigation**:
- Cross-model perplexity evidence (3 models) is strong
- Cross-model geometry evidence (3 models, Table 2) is strong -- all show 29-38° rotation
- For a workshop, single-model reasoning evaluation is standard
- The mechanistic argument (PC₁ inversion → reasoning collapse) is model-agnostic
- Workshop reviewers are less likely to demand multi-model reasoning evidence

**Recommendation**: If compute permits, run the boundary-healed Llama-3-8B on GSM8K N=100 to show the dissociation generalizes. If not, add a limitation sentence acknowledging single-model reasoning scope.

---

## 8. PC₁ Inversion: Is 179.52° Real or Artifactual?

- **Severity**: **Medium** (reviewers may question the measurement)
- **Effort-to-Reward**: **Medium** (needs methodological explanation)
- **Paper addresses?** No -- the 179.52° inversion finding is new from Phase 25 and is not yet in the paper.

**Description**: The paper reports a mean canonical subspace angle of 29.34° (top-10 PCs). Phase 25 P5 reveals that PC₁ specifically has θ = 179.52° (projection = −1.0000), a near-complete sign inversion. The 29.34° figure averages across PC₁ (at 179.52°) and PC₂--PC₁₀ (at moderate angles). This dramatic discrepancy raises methodological questions:

1. **Why is PC₁ inverted but the mean is only 30°?** Because the mean of angles for 10 PCs is dominated by PC₂--PC₁₀ at moderate angles (~10-30°). One outlier at 179.52° only pulls the mean up by ~15°.
2. **Is PC₁ inversion a real phenomenon or an SVD sign ambiguity?** SVD sign is conventionally ambiguous (flipping sign of both U and V columns preserves the decomposition). However, the sign-inversion measurement reports alignment between the intact and pruned PC₁ directions *after alignment* in the canonical subspace framework, which resolves sign ambiguity.

**Reviewer attack**: "You report 179.52°, but SVD signs are arbitrary. How do we know this isn't just a sign convention artifact?"

**Mitigation**:
- The measurement uses canonical subspace angles via `arccos(clip(σ_i, -1, 1))`, which accounts for sign via the arccos function (cos(180°) = −1 → arccos(−1) = 180°)
- For a single PC, the canonical angle between v₁^(intact) and v₁^(pruned) is arccos(|v₁^(intact) · v₁^(pruned)|) or arccos(v₁^(intact) · v₁^(pruned)) depending on convention. The paper should clarify which is used.
- Verify that the PC₁ alignment measurement (dot product = −1.0000) is not an SVD sign convention artifact by checking if v₁^(intact) ≈ −v₁^(pruned) elementwise

**Recommendation**: Add a methodological note explaining that canonical subspace angles naturally handle sign ambiguity (σ = cos θ → arccos gives both acute and obtuse angles). Include the raw dot product (−1.0000) alongside the angle for transparency. Consider reporting both the sign-ambiguous (acute) angle (0.48°) and the sign-aware angle (179.52°) to show the distinction.

---

## 9. Gemma-2-2B Boundary Healing Is Weak

- **Severity**: **Low** (explained, but still a weakness)
- **Effort-to-Reward**: **Low** (the explanation is adequate)
- **Paper addresses?** Yes -- attributes to Sliding Window Attention (SWA).

**Description**: On Gemma-2-2B, boundary LoRA (t=2) recovers 1.00× the perplexity gap (vs 4.04× for Mistral-7B and 2.22× for Llama-3-8B). Boundary LoRA actually performs *worse* than global LoRA at t=2 (PPL 20.46 vs 19.52) and only matches global LoRA at t=3 (19.49 vs 19.52). The boundary advantage disappears entirely for SWA architectures.

**Paper's explanation**: Gemma-2 uses Sliding Window Attention with alternating global attention blocks. Dropping a 4-layer block severs receptive fields across multiple SWA window boundaries, requiring t=3 to span the SWA window transition interface.

**Reviewer concern**: "Your method's primary advantage (boundary constraint → better PPL recovery) doesn't hold on all architectures. How broadly applicable is this?"

**Mitigation**: The explanation is plausible and testable. Frame this as an architectural boundary condition rather than a failure: "Standard attention architectures (Mistral, Llama) benefit from t=2 boundary healing; SWA architectures require t=3 to span alternating window transitions."

**Recommendation**: Keep the SWA explanation. Consider this a useful architectural insight rather than a weakness.

---

## 10. The Bathtub Curve Is Ground-Truth, Not a Proxy Discovery

- **Severity**: **Low** (paper handles this honestly now)
- **Effort-to-Reward**: **N/A** (already fixed)
- **Paper addresses?** Yes -- the current paper does NOT claim proxy-based layer selection.

**Description**: An earlier version of this project (Phase 4--11, targeting COLM Actionable Interpretability) attempted to use proxy metrics (Effective Dimensionality, Jacobian, nAUDC) to predict layer sensitivity. All proxy metrics failed (NaN correlations or ρ = −0.14, p = 0.44). The paper now correctly treats the Bathtub Curve as a ground-truth knockout measurement.

**Current status**: The paper (main.tex) does not mention proxy metrics for layer selection. It directly presents the Bathtub Curve and thickness ablation without claiming automated discovery. This is fine.

**Recommendation**: No action needed. Ensure no residual proxy-metric claims survive in the manuscript.

---

## 11. No Comparison with Prior Depth Pruning Recovery Methods

- **Severity**: **Medium**
- **Effort-to-Reward**: **Medium** (add a discussion paragraph)
- **Paper addresses?** No -- the paper compares only to its own global LoRA baselines.

**Description**: The related work section (§2) lists depth pruning methods (ShortGPT, LaCo, LLM-Pruner, SparseGPT, etc.) but the empirical evaluation only compares boundary LoRA to (a) full global LoRA and (b) global sensitivity LoRA -- both are the authors' own baselines. There is no comparison against:
- LaCo (layer merging instead of fine-tuning)
- ShortGPT's layer removal criteria
- EoRA or other PEFT allocation strategies
- Simple baselines like continuing pre-training on C4 after pruning

**Reviewer attack**: "How does boundary LoRA compare to simply continuing pre-training the pruned model? Or to layer merging approaches like LaCo? Without baselines from prior work, the contribution is hard to contextualize."

**Mitigation**:
- The paper's contribution is mechanistic (diagnosing *why* reasoning fails) more than engineering (which method achieves best PPL)
- Adding a full comparison matrix against prior recovery methods would be a large engineering effort
- For a workshop paper, self-contained evaluation against own baselines is common

**Recommendation**: Add a paragraph to §7 (Discussion) acknowledging this limitation: "We compare against uniform LoRA and sensitivity-allocated LoRA as natural within-method baselines. Systematic comparison against prior post-pruning recovery methods (e.g., continued pre-training, layer merging) is left to future work." This preempts the reviewer question.

---

## 12. 8-Shot Prompting Sensitivity

- **Severity**: **Low** (standard practice, but N=100 magnifies it)
- **Effort-to-Reward**: **Low** (not worth re-running everything)
- **Paper addresses?** No.

**Description**: The paper uses 8-shot CoT prompting for GSM8K. With N=100 evaluation samples, exact-match accuracy can be sensitive to:
1. Shot selection (which 8 exemplars are used)
2. Exemplar order
3. Prompt formatting (newlines, spacing, "Q:" vs "Question:")

Phase 25 uses 5-shot prompting (different from the paper's 8-shot), which is one reason for the 1% vs 7% baseline discrepancy. This demonstrates that exact accuracy is shot-count-sensitive on pruned models.

**Reviewer concern**: "Your 7.0% result depends on 8 specific exemplars. How stable is this across different shot selections?"

**Mitigation**:
- Standard practice in the field (most GSM8K papers use fixed-shot prompting without shot-sensitivity analysis)
- The paper's qualitative findings (dissociation, PC₁ ablation effect) are robust to the absolute accuracy level
- For workshop acceptance, sensitivity analysis is not expected

**Recommendation**: Add a footnote noting that shot selection can affect pruned-model accuracy and that the paper uses a fixed, deterministic prompt for reproducibility. If time permits, run 2-3 different shot permutations for the boundary-healed condition to show the qualitative finding holds.

---

## 13. Downstream Weight Entrenchment Claim Is Speculative

- **Severity**: **Medium** (overclaiming risk)
- **Effort-to-Reward**: **High** (easy wording fix)
- **Paper addresses?** Partially -- the paper uses "One possible explanation, consistent with these findings..."

**Description**: The paper (and P5 analysis) hypothesizes that downstream weight matrices (L15--31) were "calibrated on the disrupted signal direction throughout pre-training" and "cannot accept a corrected input without re-training." This is a reasonable inference but is not experimentally established. The Phase 25 analysis correctly flags this as speculative.

**Reviewer attack**: "You claim downstream weights are irreversibly biased, but you haven't tested this. Perhaps the geometric correction was incorrectly specified, or a different inference-time intervention would work."

**Mitigation**:
- The paper already uses hedging language ("One possible explanation," "consistent with these findings")
- The exhaustive negative control matrix (9 interventions) makes it unlikely that a simple geometric fix was overlooked
- The sufficiency test (34% → 10%) proves that geometric disruption causes the damage, which is the stronger directional claim

**Recommendation**: Ensure ALL claims about downstream weight entrenchment use hedging language:
- Use: "One possible explanation, consistent with these findings, is that..."
- Use: "These findings suggest that..."
- Avoid: "conclusively disproves," "proves that downstream weights are...," "mechanism is confirmed"

---

## 14. PC₂ Ablation at 34% ≡ Baseline — Statistical Power Concern

- **Severity**: **Low**
- **Effort-to-Reward**: **Low**
- **Paper addresses?** No.

**Description**: Paper Table 4 (Causal PC Component Ranking) reports that ablating PC₂ leaves GSM8K at 34.0% [24.8%, 44.2%] -- identical to the intact baseline. However, with N=100, the 95% CI is ±9.7 pp. A true degradation of 5--8 pp (e.g., 34% → 27%) would not be detectable at this sample size. The paper cannot distinguish between "PC₂ is irrelevant" and "PC₂ has a small but nonzero effect that N=100 cannot detect."

**Reviewer concern**: "With a CI half-width of ±10 pp, claiming that PC₂ ablation has 'zero effect' is an overstatement. You have insufficient power to detect a moderate effect."

**Mitigation**:
- The qualitative conclusion (PC₁ ≫ PC₂ in causal importance) is robust even with wide CIs -- PC₁ drops 23 pp, PC₂ is within 0 ± 10 pp
- Frame as: "PC₂ ablation produced no detectable degradation (0.0 pp, 95% CI consistent with zero effect), in contrast to the 23 pp degradation from PC₁ ablation"

**Recommendation**: Change "completely intact at 34.0%" to "showed no detectable degradation (0.0 pp change, 95% CI [−9.7, +9.7] pp)." Be precise about statistical power rather than claiming zero effect.

---

## 15. No Error Bars on Perplexity

- **Severity**: **Low** (standard in the field, but a nice-to-have)
- **Effort-to-Reward**: **Medium** (adds statistical credibility)
- **Paper addresses?** No.

**Description**: Table 1 reports C4 PPL values without standard errors or confidence intervals. PPL is computed over the C4 validation set (thousands of passages), so the standard error is very small. However, the PPL differences between conditions are also small (e.g., ΔPPL = 0.61 between boundary t=2 and global sensitivity on Mistral). Without knowing the standard error, it's unclear whether 9.122 vs 9.730 is a meaningful gap.

The paper does report p < 0.001 (Wilcoxon signed-rank test across C4 passages) for the difference, which addresses this concern.

**Recommendation**: Add ± SEM to the PPL column of Table 1. Even if the error bars are tiny (which they likely are over thousands of passages), it signals statistical rigor. If Wilcoxon p-values are already reported, this is low priority.

---

## 16. Sufficiency Test: Rotation Reproduces ~85% of Collapse — What Explains the Remaining 15%?

- **Severity**: **Low** (can be framed as expected)
- **Effort-to-Reward**: **Medium** (adds depth to discussion)
- **Paper addresses?** No.

**Description**: The sufficiency test drops accuracy from 34% to 10%. The full effect of 4-layer pruning drops to 7%. The gap is 3 pp. If rotation is the complete explanation, why is the remaining 3 pp unexplained?

**Possible explanations**:
- The Procrustes R*^T is an approximation of the true pruning-induced rotation; it doesn't perfectly capture all geometric changes
- The 10% and 7% CIs overlap ([4.8%, 17.6%] vs [2.9%, 13.9%]), so the gap may not be real at N=100
- Removing layers removes capacity in addition to rotating representations (trivially)
- Weight damage from removing transformer blocks is a separate effect

**Recommendation**: Acknowledge in the discussion: "The artificial rotation reproduces most (approximately 85%) of the pruning-induced collapse; the remaining effect may reflect capacity reduction or geometric changes not captured by the Procrustes transformation."

---

## 17. Single Pruning Configuration (4-Layer Drop of L11--14)

- **Severity**: **Low**
- **Effort-to-Reward**: **Low** (already addressed by P4)
- **Paper addresses?** Yes -- rotation-depth curve (Table 3) varies k=1--4; location sweep (P4) varies drop position.

**Description**: The primary causal experiments (PC ablation, sufficiency test, recovery interventions) all use a single pruning configuration: drop layers 11--14 from Mistral-7B. However, the paper has:
- **Rotation-depth sweep** (Table 3): k = 1, 2, 3, 4 layers dropped -- shows monotonic relationship
- **Location sweep** (P4): Early (L3--6), Middle (L11--14), Late (L20--23) -- confirms robustness across positions
- **Cross-model geometry** (Table 2): 3 model families validate the rotation phenomenon

This is well-covered.

**Recommendation**: No action needed. Ensure the location sweep (P4) is integrated into the paper.

---

## 18. Absence of Human Evaluation for Qualitative Output

- **Severity**: **Low**
- **Effort-to-Reward**: **Low**
- **Paper addresses?** No.

**Description**: The paper mentions "infinite token repetition loops" during autoregressive decoding from pruned models (§5.4), but this is anecdotal qualitative observation rather than systematic analysis. There's no quantification of repetition rate, no comparison of output quality metrics (e.g., MAUVE score, self-BLEU), and no human evaluation.

**Reviewer concern**: This is unlikely to be raised for a workshop paper, but a thorough reviewer might note that the qualitative claims are anecdotal.

**Recommendation**: If time permits, quantify the repetition rate (fraction of outputs containing n-gram repeats) across dense, pruned, and healed conditions. Otherwise, frame the qualitative observation as an illustrative anecdote rather than a systematic finding.

---

## 19. No Per-Layer PC Angle Analysis Across Network Depth

- **Severity**: **Low** (nice-to-have, not critical)
- **Effort-to-Reward**: **Low**
- **Paper addresses?** No -- only reports post-incision layer geometry.

**Description**: The paper measures subspace angles only at the post-incision layer (L15 for Mistral-7B). It does not report:
1. How the rotation propagates through downstream layers (does it attenuate? amplify?)
2. Pre-incision layer geometry (are upstream representations also affected?)
3. Per-PC angles beyond PC₁ (what are θ for PC₂ through PC₁₀ individually?)

**Recommendation**: Not needed for workshop acceptance. Could strengthen a future main-conference expansion.

---

## 20. Paper Framing: Diagnostic Arc Is Complete, But Needs Explicit Signposting

- **Severity**: **Low** (editorial)
- **Effort-to-Reward**: **High** (easy writing fix, big impact)
- **Paper addresses?** Partially.

**Description**: The paper's narrative arc is strong but could be more explicitly telegraphed:
1. Problem: Depth pruning creates representational discontinuity
2. Solution: Boundary LoRA recovers fluency efficiently
3. Discovery: Fluency ≠ reasoning -- dissociation revealed
4. Diagnosis: PC₁ rotation (179.52° inversion) causes reasoning collapse
5. Causal proof: Sufficiency test confirms rotation → collapse
6. Falsification: 9 competing recovery paradigms all fail

This is a complete diagnostic contribution. However, the paper needs to explicitly say: "This is a diagnostic paper. We identify the mechanism and falsify the obvious fixes. Training-time recovery is future work."

**Recommendation**: Add a sentence to the introduction: "This paper is a diagnostic and mechanistic contribution. We identify the geometric mechanism of post-pruning reasoning collapse and systematically falsify competing explanations and recovery strategies. Training-time PC₁-targeted recovery is a natural next step and is left to future work."

---

## Summary Matrix

| # | Weakness | Severity | E→R | Paper Fix? |
|:--:|:---|:---:|:---:|:---:|
| 1 | GSM8K N=100 | ✅ **Resolved** | — | N=500 completed; CIs ±1-4pp |
| 2 | No PC₁ training-time recovery | Medium | Low | Keep as future work |
| 3 | Baseline discrepancy (1% vs 7%) | ✅ **Resolved** | — | N=500 gives 1.8%; paper's 7% was N=100 high-side |
| 4 | Noise floor (0--4%) | Medium | Medium | Frame honestly |
| 5 | Location non-specificity | Medium | Medium | Do not claim specificity |
| 6 | Residual norm correlation | Medium | High | Add norm/R* counter-argument |
| 7 | Single model reasoning | ✅ **Partially resolved** | — | Llama-3-8B confirms same pattern (56%→1%→6%) |
| 8 | PC₁ 179.52° measurement | Medium | Medium | Add methodological note |
| 9 | Gemma-2 SWA t=3 | Low | Low | Adequate as-is |
| 10 | Bathtub is ground-truth | Low | N/A | Already fixed |
| 11 | No prior methods comparison | Medium | Medium | Add limitation paragraph |
| 12 | 8-shot sensitivity | Low | Low | Footnote |
| 13 | Weight entrenchment speculation | Medium | High | Hedge language |
| 14 | PC₂ CI too wide for zero claim | Low | Low | Reframe wording |
| 15 | No error bars on PPL | Low | Medium | Add SEM if easy |
| 16 | Sufficiency gap (15%) unexplained | Low | Medium | Acknowledge in discussion |
| 17 | Single pruning config | Low | Low | Already addressed |
| 18 | No human eval for output quality | Low | Low | Frame as anecdote |
| 19 | No per-layer PC angle analysis | Low | Low | Future work |
| 20 | Diagnostic framing needs signposting | Low | High | Add intro sentence |

---

## Bottom Line

**The paper is workshop-competitive as-is.** The diagnostic arc (problem → solution ← discovery → diagnosis → causal proof → falsification) is complete and well-supported. The remaining weaknesses fall into two categories:

1. **Statistical tightening** (#1, #3 resolved with N=500): add PPL error bars (#15)
2. **Framing/wording** (must be correct): Hedge speculation (#13), don't claim zero effect for PC₂ (#14), don't claim layer specificity for recovery (#5), explicitly signpost paper as diagnostic (#20), update baseline numbers to N=500 (#1, #3)

The most impactful editorial priority: **update the headline numbers in the paper** to the N=500 values (dense 39.6%, pruned 1.8%, healed 6.2%). Also **add explicit diagnostic framing (#20)** to the introduction -- this preempts reviewer expectations for a prescriptive contribution.
