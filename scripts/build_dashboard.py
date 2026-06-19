"""Generate a self-contained HTML status dashboard from experiments/*.json.

Built so a NEWCOMER or returning member can drop in cold and get oriented in
under a minute: what we're building, why, where we are, the headline result,
decisions, the task board, how we work, and what's next.
Pure stdlib + jinja2; embeds data into one HTML file (serve statically,
regenerate as runs land).

    python scripts/build_dashboard.py
"""
import os
import json
import glob
import datetime
import argparse

EXPERIMENTS = "experiments"
DASHBOARD = "dashboard"

# ---- editable narrative state ------------------------------------------------
TAGLINE = (
    "A systematic framework for diagnosing and recovering from catastrophic "
    "layer sensitivity in transformers — validated on GPT-2, scaling to "
    "Mistral-7B and Llama-3-8B."
)

OVERVIEW = [
    ("What we're building",
     "A <b>structured pruning + activation healing pipeline</b> (TIM-MGDT) that "
     "identifies which transformer layers are <i>elastic</i> vs <i>critical</i>, "
     "prunes the elastic ones, and heals the activation drift — recovering "
     "perplexity while cutting parameters, latency, and memory by ~50%."),
    ("Why this matters",
     "Existing pruning methods either destroy quality or require expensive "
     "retraining. We discovered that <b>Layer 10 in GPT-2 is elastic, not "
     "broken</b> — the +108% PPL explosion on hard removal shows over-weighting, "
     "not architectural damage. A 2% scaling fix restores balance. TIM-MGDT "
     "exploits this insight systematically."),
    ("Where we are today",
     "<b>Phase 11 complete</b> — real GPU validation on GPT-2 (124M). Found a "
     "<b>30% sparsity Winning Ticket</b> (+4.9% PPL only). Discovered that "
     "activation healing <i>hurts</i> GPT-2 (Conv1D architecture) but will "
     "<b>massively help</b> Llama/Mistral (nn.Linear). Next: 7B model runs on "
     "dual 3090s."),
]

RESEARCH_Q = (
    "The research question",
    "Can <b>temporal importance metrics</b> — cheap proxy signals computed from "
    "gradient dynamics — reliably <b>identify and exploit</b> layer-level "
    "elasticity to achieve lossless (< 5% PPL) structured compression at "
    "<b>50%+ parameter reduction</b> without retraining?"
)

PHASE = (
    "Phase 11 complete (GPT-2 real GPU validation). "
    "Winning Ticket found at 30% sparsity. "
    "Next: Llama-3-8B and Mistral-7B on dual 3090 remote server."
)

RUNS = [
    ("Phase 2 — Ground Truth sensitivity mapping (GPT-2)", "done"),
    ("Phase 5.1 — Hard pruning baseline (expected fail: +108% PPL)", "done"),
    ("Phase 5.2 — Soft rebalancing (+0.31% PPL recovery)", "done"),
    ("Phase 5.3 — Progressive optimization (0.98x optimal scaling)", "done"),
    ("Phase 6 — 500-sample validation (-0.11% PPL improvement)", "done"),
    ("Phase 10 — Lottery ticket pipeline (GPU metrics recorded)", "done"),
    ("Phase 11 — Real pruning & measurement (GPT-2, 15x GPU speedup)", "done"),
    ("Phase 11 — Llama-3-8B real evaluation (dual 3090)", "queued"),
    ("Phase 11 — Mistral-7B real evaluation (dual 3090)", "queued"),
]

DECISIONS = [
    ("Architecture insight", "Layer 10 is elastic, not broken",
     "+108% PPL on hard removal = over-weighting. 2% scaling restores it. "
     "This insight drives the entire compression strategy."),
    ("Winning Ticket", "30% unstructured sparsity (GPT-2)",
     "+4.9% PPL only. Above 30%, PPL degrades rapidly (+21.3% at 40%) — "
     "classic phase transition boundary verified."),
    ("Activation Healing", "Conv1D blocks immune → nn.Linear blocks will benefit massively",
     "GPT-2 intermediate layers use Conv1D (unpruned by global linear pruning). "
     "Healing added noise. Llama/Mistral use nn.Linear everywhere — healing "
     "will recover massive quality loss."),
    ("Head Pruning", "MHA is destructive; GQA will be far more tolerant",
     "GPT-2 MHA: +59.3% PPL at 20% head pruning. Llama/Mistral GQA (8 KV groups "
     "for 32 Q heads) has built-in query redundancy → much safer."),
    ("Eval metric", "Perplexity on WikiText-2 (PPL)",
     "Standard LM evaluation. Spearman ρ > 0.7 for proxy-to-ground-truth "
     "correlation. P-value < 0.05 for statistical significance."),
    ("Compute", "Apple Silicon (quick runs) → 2×3090 (48 GB, full runs)",
     "Two-phase compute strategy. Remote SSH to dual 3090 for 7B models. "
     "See REMOTE_SETUP_GUIDE.md."),
]

TASKS = [
    ("Saksham", "GPU pipeline", "active",
     "Phase 11 GPT-2 validation complete. NEXT: (1) launch "
     "runners/run_llama_phase11.sh on remote 3090, (2) launch "
     "runners/run_mistral_phase11.sh, (3) analyze causal patching curves "
     "and activation healing recovery on 7B models."),
    ("Saksham", "Analysis & paper", "active",
     "Phase 11 GPT-2 results analyzed. NEXT: (1) interpret 7B causal patching "
     "bell curves, (2) write up Conv1D vs nn.Linear healing discovery, "
     "(3) compare IMP winning ticket sparsities across model scales."),
    ("Saksham", "Dashboard & infrastructure", "done",
     "Automated dashboard deployed via GitHub Pages + GitHub Actions. "
     "state.json exposed for agent/MCP consumption."),
]

HOW_WE_WORK = [
    ("Cadence", "Solo project · ~10+ hrs/week · structured sprints"),
    ("Compute", "Two-phase: <b>Apple Silicon</b> (M-series, quick) → "
     "<b>2×3090 (48 GB VRAM)</b> via SSH for 7B models"),
    ("Code", "<b>GitHub only</b> — single repo "
     "<code>SakshamKapoor2911/tim-mgdt</code>, frequent commits, "
     "no Colab sprawl"),
    ("Method", "Structured pruning · causal activation patching · "
     "forward-hook healing · IMP winning tickets · real GPU measurement"),
]

BULLETPROOFING = [
    ("done", "Phase 6 validation on 500-sample WikiText-2 — -0.11% PPL improvement confirmed."),
    ("done", "Phase 5.1 hard pruning baseline — +108% PPL explosion (expected) establishes failure mode."),
    ("done", "Phase 11 GPT-2 winning ticket — 30% sparsity, +4.9% PPL only, verified on real GPU."),
    ("done", "PyTorch reparameterization quirk documented — weight_orig vs weight after remove_global_pruning."),
    ("done", "Conv1D vs nn.Linear healing behavior explained — GPT-2 is architectural edge case."),
    ("todo", "Llama-3-8B causal patching — expect bell-shaped recovery curve (layers 12-24 critical)."),
    ("todo", "Mistral-7B GQA head pruning — expect much higher tolerance than GPT-2 MHA."),
    ("todo", "Activation healing on 7B — expect massive PPL recovery since all layers are nn.Linear."),
    ("idea", "InAttention integration with winning-ticket masks for real hardware latency reduction."),
    ("idea", "KL divergence proxy for 100x profiling speedup on 7B models."),
]

MILESTONES = [
    ("done", "Phases 2-6 complete: ground truth mapping → validation on GPT-2"),
    ("done", "Phase 10: lottery ticket pipeline built, GPU metrics recorded"),
    ("done", "Phase 11 GPT-2: real pruning, 30% winning ticket, healing discovery"),
    ("done", "Automated dashboard + CI/CD deployment on GitHub Pages"),
    ("now", "Phase 11 Llama-3 & Mistral-7B: real 7B evaluation on dual 3090s"),
    ("then", "Phases 7-9: alternative corpora, downstream tasks, paper"),
]

PATH_TO_PAPER = [
    ("done", "Saksham", "Phases 2-6 complete; elasticity discovery validated on GPT-2."),
    ("done", "Saksham", "Phase 11 GPT-2 real pruning — all 4 stages verified on GPU."),
    ("done", "Saksham", "Phase 11 analysis written — Conv1D/nn.Linear discovery, translation predictions."),
    ("now", "Saksham", "Launch 7B runs: runners/run_llama_phase11.sh and runners/run_mistral_phase11.sh on remote 3090."),
    ("now", "Saksham", "Design 7B analysis framework — causal patching curves, healing recovery rates."),
    ("next", "Saksham", "InAttention integration with winning-ticket masks for real latency benchmarks."),
    ("next", "Saksham", "Phase 7 alternative corpora evaluation; begin paper draft."),
    ("then", "Saksham", "Submit — workshop or conference. Quality over timeline."),
]

LINKS = [
    ("Repository", "https://github.com/SakshamKapoor2911/tim-mgdt"),
    ("Phase 11 Analysis", "https://github.com/SakshamKapoor2911/tim-mgdt/blob/master/experiments/phase11/gpt2_real_results_analysis.md"),
    ("Remote Setup Guide", "https://github.com/SakshamKapoor2911/tim-mgdt/blob/master/REMOTE_SETUP_GUIDE.md"),
    ("Phase Status Ledger", "https://github.com/SakshamKapoor2911/tim-mgdt/blob/master/PHASE_STATUS_LEDGER.md"),
]

LITERATURE = [
    ("All Circuits Lead to Rome", "Chen et al., 2026", "Shows LLM tasks are supported by multiple, structurally distinct redundant circuits. <b>Our diff</b>: This theoretical redundancy explains the physical 'elasticity' we exploit; we systematically prune these redundant pathways and use activation healing to force the model onto the remaining circuits."),
    ("The Lottery Ticket Hypothesis", "Frankle & Carbin, 2019", "Dense networks contain sparse subnetworks that can match original accuracy if trained from scratch. <b>Our diff</b>: We find winning tickets <i>post-pretraining</i> in massive LLMs using gradient dynamics, entirely avoiding the retraining phase."),
    ("SparseGPT", "Frantar et al., 2023", "One-shot pruning using Hessian inverse. High VRAM requirement. <b>Our diff</b>: We use cheap 1st-order gradient dynamics instead of expensive 2nd-order Hessians."),
    ("Wanda", "Sun et al., 2023", "Prunes based on weight magnitude × input activation norm. <b>Our diff</b>: We use temporal gradient variance to find elasticity, avoiding need for large calibration sets."),
    ("SliceGPT", "Ashkboos et al., 2024", "Deletes entire rows/cols and uses PCA rotation. <b>Our diff</b>: We prune without destroying the original parameter space mapping, enabling forward-hook healing."),
    ("ShortGPT", "Men et al., 2024", "Removes entire blocks based on block-level redundancy. <b>Our diff</b>: We do fine-grained structured/unstructured pruning, rather than taking out whole transformer blocks."),
]
# ------------------------------------------------------------------------------


def _load(path):
    """Safely load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _pill(state):
    return f"<span class='pill p-{state}'>{state}</span>"


def _gap_badge(g):
    if g is None:
        return "<span class='muted'>—</span>"
    cls = "pos" if g > 0.02 else ("neg" if g < -0.02 else "zero")
    return f"<span class='b b-{cls}'>{g:+.3f}</span>"


def gather_experiment_data():
    """Scan experiments/ for JSON results, focused on latest phases."""
    results = {}

    # Dynamically find the newest complete Phase 11 run (GPT-2, Llama, or Mistral)
    phase11_files = glob.glob(os.path.join(EXPERIMENTS, "phase11", "*_complete_results.json"))
    if phase11_files:
        latest_file = max(phase11_files, key=os.path.getmtime)
        results["phase11"] = _load(latest_file)
        # Extract model name for the UI (e.g., gpt2, mistral7b, llama3)
        basename = os.path.basename(latest_file)
        results["_latest_model_name"] = basename.split("_")[0].upper()
    else:
        # Fallback to basic results if no complete run exists
        basic_files = glob.glob(os.path.join(EXPERIMENTS, "phase11", "*_results.json"))
        if basic_files:
            latest_file = max(basic_files, key=os.path.getmtime)
            results["phase11"] = _load(latest_file)
            results["_latest_model_name"] = os.path.basename(latest_file).split("_")[0].upper()

    # Phase 10
    for f in glob.glob(os.path.join(EXPERIMENTS, "phase10", "*.json")):
        key = os.path.basename(f).replace(".json", "")
        results[f"phase10_{key}"] = _load(f)

    # Count total experiments
    all_json = glob.glob(os.path.join(EXPERIMENTS, "**", "*.json"), recursive=True)
    results["_total_experiments"] = len(all_json)

    return results


def extract_metrics(data):
    """Extract deployment metrics from phase 11 results."""
    if not data:
        return None
    try:
        stages = data.get("stages", {})

        # Stage 2: IMP search
        s2 = stages.get("stage2_imp_search", {})
        winner = s2.get("best_winner", {})

        # Stage 3: Causal patching
        s3 = stages.get("stage3_causal_patching", {})
        critical = s3.get("critical_layers", [])
        safe = s3.get("safe_to_prune_layers", [])

        # Stage 4: Activation healing
        s4 = stages.get("stage4_activation_healing", {})
        deploy = s4.get("deployment_metrics", {})

        return {
            "winning_sparsity": winner.get("sparsity", 0),
            "ppl_loss_pct": winner.get("ppl_loss_increase_pct", 0),
            "critical_layers": critical[:5],
            "safe_layers": safe[:5],
            "covariance_restoration": s4.get("covariance_restoration_score", 0),
            "layers_healed": s4.get("layers_healed_count", 0),
            "baseline_params_b": deploy.get("baseline_parameters_b", 0),
            "healed_params_b": deploy.get("healed_parameters_b", 0),
            "baseline_latency_ms": deploy.get("baseline_latency_ms", 0),
            "healed_latency_ms": deploy.get("healed_latency_ms", 0),
            "baseline_memory_gb": deploy.get("baseline_memory_gb", 0),
            "healed_memory_gb": deploy.get("healed_memory_gb", 0),
            "perplexity_recovery_pct": deploy.get("perplexity_recovery_pct", 0),
        }
    except Exception:
        return None


def build_state_json(experiments, metrics):
    """Build the machine-readable state.json for agent consumption."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        "project": "TIM-MGDT",
        "tagline": TAGLINE,
        "current_phase": "Phase 11",
        "phase_status": PHASE,
        "total_experiments": experiments.get("_total_experiments", 0),
        "latest_run": {
            "name": f"{experiments.get('_latest_model_name', 'GPT-2')} Phase 11 (Real GPU)",
            "status": "complete" if experiments.get("phase11") else "pending",
            "model": experiments.get("_latest_model_name", "gpt2").lower(),
            "device": experiments.get("phase11", {}).get("device", "unknown") if experiments.get("phase11") else "unknown",
        },
        "metrics": metrics,
        "research_question": RESEARCH_Q[1].replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""),
        "overview": [{"title": t, "body": b.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", "")} for t, b in OVERVIEW],
        "decisions": [{"topic": t, "choice": c, "rationale": r} for t, c, r in DECISIONS],
        "tasks": [{"who": w, "track": tr, "status": s, "next_action": a} for w, tr, s, a in TASKS],
        "milestones": [{"status": s, "description": d} for s, d in MILESTONES],
        "bulletproofing": [{"status": s, "item": i} for s, i in BULLETPROOFING],
        "path_to_paper": [{"status": s, "owner": o, "item": i} for s, o, i in PATH_TO_PAPER],
        "runs": [{"label": l, "state": s} for l, s in RUNS],
        "links": {name: url for name, url in LINKS},
    }


# ---- HTML generation (self-contained, no Jinja2 dependency for CI) -----------

def _overview_cards():
    cards = ""
    for title, body in OVERVIEW:
        cards += f"""<div class="ov-card"><h3>{title}</h3><p>{body}</p></div>\n"""
    return cards


def _research_callout():
    return f"""<div class="callout"><h3>{RESEARCH_Q[0]}</h3><p>{RESEARCH_Q[1]}</p></div>"""


def _runs_html():
    rows = ""
    for label, state in RUNS:
        rows += f"<tr><td>{label}</td><td>{_pill(state)}</td></tr>\n"
    return f"""
<div class="section-label">Experiment runs</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="data-table">
<tr><th>Run</th><th>Status</th></tr>
{rows}
</table></div>"""


def _decisions_html():
    rows = ""
    for topic, choice, rationale in DECISIONS:
        rows += f"""<tr><td class="task-who">{topic}</td><td><b>{choice}</b></td>
<td class="task-action">{rationale}</td></tr>\n"""
    return f"""
<div class="section-label">Key decisions</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="task-table">
<tr><th>Topic</th><th>Decision</th><th>Rationale</th></tr>
{rows}
</table></div>"""


def _metrics_html(metrics, model_name="GPT-2"):
    if not metrics:
        return "<div class='card'><p class='muted'>No metrics available yet.</p></div>"

    def pct_change(base, opt):
        if base == 0:
            return "—"
        return f"-{100*(1 - opt/base):.1f}%"

    return f"""
<div class="section-label">Performance vs Efficiency — Phase 11 ({model_name})</div>
<div class="g2">
<div class="result-card primary">
  <div class="rc-head"><h3>Deployment Metrics</h3>
  <div class="rc-meta">{model_name} · Real GPU measurement</div></div>
  <table class="cond-table">
    <tr><td>Parameters</td><td class="val">{metrics['baseline_params_b']:.2f}B → {metrics['healed_params_b']:.2f}B</td></tr>
    <tr><td>Latency</td><td class="val">{metrics['baseline_latency_ms']:.1f}ms → {metrics['healed_latency_ms']:.1f}ms</td></tr>
    <tr><td>Memory</td><td class="val">{metrics['baseline_memory_gb']:.2f}GB → {metrics['healed_memory_gb']:.2f}GB</td></tr>
    <tr class="winner"><td>Perplexity Recovery</td><td class="val">{metrics['perplexity_recovery_pct']:.1f}%</td></tr>
  </table>
  <div class="metric-row">
    <div class="metric-cell"><span class="metric-label">Param Reduction</span>
    <span class="metric-value good">{pct_change(metrics['baseline_params_b'], metrics['healed_params_b'])}</span></div>
    <div class="metric-cell"><span class="metric-label">Latency Reduction</span>
    <span class="metric-value good">{pct_change(metrics['baseline_latency_ms'], metrics['healed_latency_ms'])}</span></div>
  </div>
</div>
<div class="result-card">
  <div class="rc-head"><h3>IMP Winning Ticket</h3>
  <div class="rc-meta">Iterative Magnitude Pruning · Stage 2</div></div>
  <table class="cond-table">
    <tr class="winner"><td>Best Sparsity</td><td class="val">{metrics['winning_sparsity']*100:.0f}%</td></tr>
    <tr><td>PPL Loss</td><td class="val">+{metrics['ppl_loss_pct']*100:.2f}%</td></tr>
    <tr><td>Layers Healed</td><td class="val">{metrics['layers_healed']}</td></tr>
    <tr><td>Covariance Restoration</td><td class="val">{metrics['covariance_restoration']:.3f}</td></tr>
  </table>
  <div class="rc-note">
    <b>Critical layers:</b> {', '.join(metrics['critical_layers'][:5])}<br>
    <b>Safe to prune:</b> {', '.join(metrics['safe_layers'][:5])}
  </div>
</div>
</div>"""


def _tasks_html():
    rows = ""
    for who, track, status, action in TASKS:
        rows += f"""<tr><td class="task-who">{who}<br><span class="task-track">{track}</span></td>
<td>{_pill(status)}</td><td class="task-action">{action}</td></tr>\n"""
    return f"""
<div class="section-label">Task board</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="task-table">
<tr><th>Who</th><th>Status</th><th>Next action</th></tr>
{rows}
</table></div>"""


def _how_we_work():
    items = ""
    for title, body in HOW_WE_WORK:
        items += f"<div class='card'><div class='card-title'>{title}</div><p>{body}</p></div>\n"
    return f"""<div class="section-label">How we work</div><div class="g2">{items}</div>"""


def _bulletproofing_html():
    rows = ""
    for status, item in BULLETPROOFING:
        rows += f"<tr><td>{_pill(status)}</td><td>{item}</td></tr>\n"
    return f"""
<div class="section-label">Bulletproofing checklist</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="data-table">
<tr><th>Status</th><th>Item</th></tr>
{rows}
</table></div>"""


def _milestones_html():
    items = ""
    for status, desc in MILESTONES:
        items += f"""<div class="mile"><div>{_pill(status)}</div><div style="font-size:13px;color:var(--ink2)">{desc}</div></div>\n"""
    return f"""<div class="section-label">Milestones</div><div class="milestones">{items}</div>"""


def _path_to_paper():
    rows = ""
    for status, owner, item in PATH_TO_PAPER:
        rows += f"<tr><td>{_pill(status)}</td><td class='task-who'>{owner}</td><td class='task-action'>{item}</td></tr>\n"
    return f"""
<div class="section-label">Path to paper</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="task-table">
<tr><th>Status</th><th>Owner</th><th>Item</th></tr>
{rows}
</table></div>"""


def _literature_html():
    rows = ""
    for paper, authors, notes in LITERATURE:
        rows += f"<tr><td class='task-who'>{paper}<br><span class='task-track'>{authors}</span></td><td class='task-action'>{notes}</td></tr>\n"
    return f"""
<div class="section-label">Literature & Related Work</div>
<div class="card" style="padding:0;overflow:hidden">
<table class="task-table">
<tr><th>Paper / Authors</th><th>Context & Our Differentiation</th></tr>
{rows}
</table></div>"""


def _links_html():
    items = ""
    for name, url in LINKS:
        items += f'<a class="nav-gh" href="{url}" target="_blank">{name}</a>\n'
    return f"""<div class="section-label">Links</div><div style="display:flex;flex-wrap:wrap;gap:8px">{items}</div>"""


CSS = """:root{--bg:#0a0c10;--bg2:#0d1117;--surface:#111620;--surface2:#161c28;--surface3:#1c2436;--border:#1f2a3e;--border2:#2a374f;--ink:#e8edf5;--ink2:#a8b8cc;--ink3:#5e7490;--accent:#4a7cf5;--accent-dim:rgba(74,124,245,.10);--accent-glow:rgba(74,124,245,.04);--green:#2ea86a;--green-dim:rgba(46,168,106,.10);--red:#d95f5f;--red-dim:rgba(217,95,95,.10);--yellow:#c4972a;--yellow-dim:rgba(196,151,42,.10);--purple:#8b74d4;--purple-dim:rgba(139,116,212,.10);--r:8px;--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--mono:'JetBrains Mono','SF Mono','Cascadia Code',Consolas,monospace}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--font);font-size:14px;line-height:1.65;color:var(--ink2);background:var(--bg);-webkit-font-smoothing:antialiased}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
nav{position:sticky;top:0;z-index:100;height:52px;display:flex;align-items:center;gap:4px;padding:0 28px;background:rgba(10,12,16,.85);backdrop-filter:blur(20px) saturate(180%);border-bottom:1px solid var(--border)}
.nav-brand{font-weight:700;font-size:15px;color:var(--ink);letter-spacing:-.3px;margin-right:20px;white-space:nowrap;display:flex;align-items:center;gap:8px}
.nav-dot{width:7px;height:7px;border-radius:50%;background:var(--green);opacity:.85}
.nav-links{display:flex;gap:2px}
.nav-links a{color:var(--ink3);font-size:13px;text-decoration:none;padding:5px 10px;border-radius:6px;transition:color .15s,background .15s}
.nav-links a:hover{color:var(--ink);background:var(--surface2)}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:12px}
.nav-meta{font-size:12px;color:var(--ink3)}
.nav-gh{font-size:12.5px;color:var(--ink3);text-decoration:none;border:1px solid var(--border2);padding:4px 12px;border-radius:6px;transition:all .15s;font-weight:500}
.nav-gh:hover{color:var(--ink);border-color:var(--accent)}
.hero{position:relative;overflow:hidden;padding:44px 36px 36px;border-bottom:1px solid var(--border);background:var(--bg2)}
.hero-inner{max-width:1120px;margin:0 auto}
.hero-eyebrow{font-size:11px;font-weight:600;letter-spacing:1.4px;text-transform:uppercase;color:var(--ink3);margin-bottom:12px}
.hero h1{font-size:26px;font-weight:700;color:var(--ink);letter-spacing:-.3px;line-height:1.25;margin-bottom:10px}
.hero-sub{font-size:15px;color:var(--ink2);max-width:640px;line-height:1.7;margin-bottom:24px}
.hero-chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:600;letter-spacing:.2px;border:1px solid transparent}
.chip-green{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.chip-blue{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.chip-yellow{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.chip-gray{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px 80px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--ink3);display:flex;align-items:center;gap:12px;margin:44px 0 18px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px}
.card-title{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:6px}
.ov-card{background:var(--surface);border:1px solid var(--border);border-top:2px solid var(--accent);border-radius:var(--r);padding:20px 22px}
.ov-card h3{font-size:13.5px;font-weight:600;color:var(--ink);margin-bottom:10px}
.ov-card p{font-size:13px;color:var(--ink2);line-height:1.65}
.ov-card strong{color:var(--ink);font-weight:600}
.callout{background:var(--accent-glow);border:1px solid rgba(74,124,245,.2);border-left:3px solid var(--accent);border-radius:var(--r);padding:18px 22px}
.callout h3{font-size:13px;font-weight:600;color:var(--accent);margin-bottom:8px}
.callout p{font-size:13.5px;color:var(--ink2);line-height:1.7}
.callout strong{color:var(--ink)}
.result-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;display:flex;flex-direction:column}
.result-card.primary{border-color:rgba(74,124,245,.4);box-shadow:0 0 0 1px rgba(74,124,245,.1),0 8px 32px rgba(74,124,245,.06)}
.rc-head{padding:16px 20px 14px;border-bottom:1px solid var(--border);background:var(--surface2)}
.rc-head h3{font-size:13.5px;font-weight:600;color:var(--ink)}
.rc-head .rc-meta{font-size:12px;color:var(--ink3);margin-top:2px}
.result-card.primary .rc-head{background:linear-gradient(135deg,rgba(74,124,245,.08) 0%,var(--surface2) 100%)}
.cond-table{width:100%}
.cond-table tr{border-bottom:1px solid var(--border)}
.cond-table tr:last-child{border-bottom:none}
.cond-table td{padding:10px 18px;font-size:13px;color:var(--ink2)}
.cond-table td:first-child{color:var(--ink3);font-size:12.5px}
.cond-table td.val{font-family:var(--mono);font-size:13.5px;font-weight:500;color:var(--ink);text-align:right}
.cond-table tr.winner td{background:rgba(46,168,106,.05)}
.cond-table tr.winner td.val{color:var(--green)}
.metric-row{display:grid;grid-template-columns:repeat(2,1fr);gap:0;border-top:1px solid var(--border)}
.metric-cell{padding:14px 18px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:3px}
.metric-cell:last-child{border-right:none}
.metric-label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);font-weight:600}
.metric-value{font-family:var(--mono);font-size:22px;font-weight:500;color:var(--ink);line-height:1.1}
.metric-value.good{color:var(--green)}
.rc-note{padding:10px 18px 14px;font-size:12px;color:var(--ink3);line-height:1.55;border-top:1px solid var(--border)}
.b{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;font-family:var(--mono)}
.b-pos{background:var(--green-dim);color:var(--green)}
.b-neg{background:var(--red-dim);color:var(--red)}
.b-zero{background:var(--surface2);color:var(--ink3)}
.data-table{width:100%;border-collapse:collapse;font-size:13px}
.data-table th{text-align:left;padding:10px 16px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);border-bottom:1px solid var(--border);background:var(--surface2)}
.data-table td{padding:11px 16px;color:var(--ink2);border-bottom:1px solid var(--border);vertical-align:middle}
.data-table tr:last-child td{border-bottom:none}
.data-table tr:hover td{background:rgba(255,255,255,.025)}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;white-space:nowrap;border:1px solid transparent}
.p-done{background:var(--green-dim);color:var(--green);border-color:rgba(46,168,106,.25)}
.p-running,.p-active{background:var(--accent-dim);color:var(--accent);border-color:rgba(74,124,245,.25)}
.p-queued,.p-todo,.p-next,.p-then{background:var(--surface2);color:var(--ink3);border-color:var(--border2)}
.p-idea{background:var(--purple-dim);color:var(--purple);border-color:rgba(139,116,212,.25)}
.p-now{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(196,151,42,.25)}
.task-table{width:100%;border-collapse:collapse}
.task-table th{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);text-align:left;padding:10px 18px;border-bottom:1px solid var(--border);background:var(--surface2)}
.task-table td{padding:14px 18px;border-bottom:1px solid var(--border);vertical-align:top}
.task-table tr:last-child td{border-bottom:none}
.task-table tr:hover td{background:rgba(255,255,255,.018)}
.task-who{font-weight:600;color:var(--ink);white-space:nowrap;font-size:13.5px}
.task-track{font-size:12.5px;color:var(--ink3)}
.task-action{font-size:13px;color:var(--ink2);line-height:1.6}
.milestones{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start}
.mile{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:12px 14px;display:flex;flex-direction:column;gap:6px}
.muted{color:var(--ink3)}
@media(max-width:768px){.g2,.g3,.milestones{grid-template-columns:1fr}}
"""


def build_html(experiments, metrics):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = experiments.get("_total_experiments", 0)
    done_count = sum(1 for _, s in RUNS if s == "done")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TIM-MGDT — project dashboard</title>
<style>{CSS}</style>
</head>
<body>
<nav>
  <div class="nav-brand"><span class="nav-dot"></span> TIM-MGDT</div>
  <div class="nav-links">
    <a href="#overview">Overview</a>
    <a href="#results">Results</a>
    <a href="#decisions">Decisions</a>
    <a href="#tasks">Tasks</a>
    <a href="#milestones">Milestones</a>
    <a href="#path">Path to Paper</a>
    <a href="#literature">Literature</a>
  </div>
  <div class="nav-right">
    <span class="nav-meta">{now}</span>
    <a class="nav-gh" href="https://github.com/SakshamKapoor2911/tim-mgdt" target="_blank">GitHub</a>
    <a class="nav-gh" href="state.json">state.json</a>
  </div>
</nav>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-eyebrow">Temporal Importance Metrics for Model Gradient-based Diagnostic Tuning</div>
    <h1>TIM-MGDT Project Dashboard</h1>
    <p class="hero-sub">{TAGLINE}</p>
    <div class="hero-chips">
      <span class="chip chip-green">Phase 11 complete</span>
      <span class="chip chip-blue">{done_count}/{len(RUNS)} runs done</span>
      <span class="chip chip-yellow">{total} experiment files</span>
      <span class="chip chip-gray">GPT-2 · Llama-3 · Mistral-7B</span>
    </div>
  </div>
</div>

<div class="wrap">
  <div class="section-label" id="overview">Project overview</div>
  <div class="g3">
    {_overview_cards()}
  </div>
  <div style="margin-top:16px">{_research_callout()}</div>

  <div id="results">{_metrics_html(metrics, experiments.get('_latest_model_name', 'GPT-2'))}</div>
  {_runs_html()}
  <div id="decisions">{_decisions_html()}</div>
  <div id="tasks">{_tasks_html()}</div>
  {_how_we_work()}
  {_bulletproofing_html()}
  <div id="milestones">{_milestones_html()}</div>
  <div id="path">{_path_to_paper()}</div>
  <div id="literature">{_literature_html()}</div>
  {_links_html()}
</div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Build TIM-MGDT dashboard")
    parser.add_argument("--out-html", default=os.path.join(DASHBOARD, "index.html"))
    parser.add_argument("--out-json", default=os.path.join(DASHBOARD, "state.json"))
    args = parser.parse_args()

    os.makedirs(DASHBOARD, exist_ok=True)

    # 1. Gather experiment data
    experiments = gather_experiment_data()
    metrics = extract_metrics(experiments.get("phase11"))

    # 2. Write state.json (agent-readable)
    state = build_state_json(experiments, metrics)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"Wrote {args.out_json}")

    # 3. Write index.html (human-readable)
    html = build_html(experiments, metrics)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
