# Phase 19 Workshop Target

Current target: COLM 2026 Workshop on Efficient Reasoning.

## Official Facts Checked July 6, 2026

- Workshop: The 2nd Workshop on Efficient Reasoning, COLM 2026.
- Date and mode listed by workshop: October 9, 2026, hybrid.
- Submission deadline: July 19, 2026 AoE.
- Notification: July 31, 2026 AoE.
- Format: single PDF in COLM format, 4-10 pages main text, references excluded.
- Reviewing: double-blind; linked supplementary material and code must also preserve anonymity.
- Contact: er.colm2026@gmail.com.
- COLM main venue: Hilton Union Square, San Francisco, October 6-9, 2026.
- COLM FAQ says the main conference is in-person only, and accepted papers must be presented in person. If no author can attend, a colleague may present only after other options are exhausted.

Sources:

- https://wdlctc.github.io/efficient-reasoning-2026/
- https://colmweb.org/
- https://colmweb.org/faq.html

## Fit

Phase 19 directly matches the workshop scope:

- Compression/pruning for efficient reasoning.
- Empirical investigation of reasoning accuracy under efficiency constraints.
- Safety/robustness-adjacent question: compression can preserve perplexity while damaging reasoning behavior.
- Systems motivation: distinguish diagnostic dense-shaped masks from hardware-relevant structural pruning.

Current strongest claim:

- Dense Mistral-7B full GSM8K: 38.67% strict EM.
- 20% dense-shaped mask full GSM8K: 37.23% strict EM.
- Delta: -1.44 points.
- Chain-length sanity: 20% mask average generation length is 57.05 words versus 56.58 for dense.

Current caveat:

- 30% and 40% mask full GSM8K are still running.
- Phase 20 mask-guided structured pruning is queued as the hardware-relevant follow-up.

## Presentation Constraint

Because the user is a US-based F-1 student and prefers venues that allow virtual presentation, do not assume COLM ER is acceptable until the workshop organizers confirm the contributed-paper presentation policy. The workshop page says hybrid, but the COLM FAQ says the main conference is in-person only and papers must be presented in person.

Recommended email:

```text
Subject: Remote presentation policy for accepted workshop papers

Hi Efficient Reasoning organizers,

I am preparing a submission for the COLM 2026 Efficient Reasoning workshop. The workshop page lists the event as hybrid, and I wanted to confirm the author presentation policy before submission.

If a paper is accepted, may an author present the contributed talk/poster remotely, or is in-person presentation by an author required? I am a full-time student in the US and may have travel constraints, so I want to make sure I only submit if I can satisfy the presentation requirements.

Thanks,
Saksham Kapoor
```

## Next Decision Gate

1. If 30% mask full GSM8K stays within about 3 points of dense, frame the paper as a broad dense-shaped mask retention regime.
2. If 30% drops but 20% remains strong, frame the paper as a thresholded structural-geometry dissociation.
3. If Phase 20 finds any mask-guided structured point that beats uniform structural pruning at similar sparsity, elevate it from diagnostic workshop paper to method paper.
4. If organizers do not allow remote presentation or a workable proxy, keep COLM ER as a technical fit but prioritize another hybrid/remote-friendly workshop.
