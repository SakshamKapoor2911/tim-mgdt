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
from pathlib import Path

EXPERIMENTS = "experiments"
DASHBOARD = "dashboard"

CONFIG_FILE = "scripts/dashboard_config.json"

try:
    with open(CONFIG_FILE, 'r') as f:
        _cfg = json.load(f)
except Exception as e:
    print(f"Error loading {CONFIG_FILE}: {e}")
    _cfg = {}

TAGLINE = _cfg.get("TAGLINE", "")
CURRENT_PHASE = _cfg.get("CURRENT_PHASE", "Phase 18B")
OVERVIEW = _cfg.get("OVERVIEW", [])
RESEARCH_Q = _cfg.get("RESEARCH_Q", ["", ""])
PHASE = _cfg.get("PHASE", "")
RUNS = _cfg.get("RUNS", [])
DECISIONS = _cfg.get("DECISIONS", [])
TASKS = _cfg.get("TASKS", [])
HOW_WE_WORK = _cfg.get("HOW_WE_WORK", [])
BULLETPROOFING = _cfg.get("BULLETPROOFING", [])
MILESTONES = _cfg.get("MILESTONES", [])
PATH_TO_PAPER = _cfg.get("PATH_TO_PAPER", [])
LINKS = _cfg.get("LINKS", [])
LITERATURE = _cfg.get("LITERATURE", [])
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
    if state.startswith("done"):
        cls = "done"
    elif state in {"planned", "todo"}:
        cls = "queued"
    elif state == "today":
        cls = "now"
    else:
        cls = state
    return f"<span class='pill p-{cls}'>{state}</span>"


def _gap_badge(g):
    if g is None:
        return "<span class='muted'>—</span>"
    cls = "pos" if g > 0.02 else ("neg" if g < -0.02 else "zero")
    return f"<span class='b b-{cls}'>{g:+.3f}</span>"


def _phase11_model_label(path, data):
    text = " ".join(
        [
            os.path.basename(path).lower(),
            str((data or {}).get("model_key", "")).lower(),
            str((data or {}).get("hf_id", "")).lower(),
        ]
    )
    if "mistral" in text:
        return "Mistral-7B"
    if "llama" in text:
        return "Llama-3-8B"
    return os.path.basename(path).split("_phase11")[0].replace("_", "-").upper()


def _phase11_priority(path, data):
    label = _phase11_model_label(path, data).lower()
    if "mistral" in label:
        return 0
    if "llama" in label:
        return 1
    if "gpt" in label:
        return 10
    return 5


def gather_experiment_data():
    """Scan experiments/ for JSON results, focused on latest phases."""
    results = {}

    # Prefer modern Mistral/Llama GPU baselines, then pick the newest run
    # within the selected model family.
    phase11_files = sorted(
        set(
            glob.glob(os.path.join(EXPERIMENTS, "phase11", "*_complete_results.json"))
            + glob.glob(os.path.join(EXPERIMENTS, "phase11", "*_results.json"))
        )
    )
    phase11_candidates = []
    for path in phase11_files:
        data = _load(path)
        if not data:
            continue
        phase11_candidates.append(
            (
                _phase11_priority(path, data),
                -os.path.getmtime(path),
                path,
                data,
                _phase11_model_label(path, data),
            )
        )
    if phase11_candidates:
        _, _, latest_file, latest_data, latest_label = sorted(phase11_candidates)[0]
        results["phase11"] = latest_data
        results["_latest_model_name"] = latest_label
        results["_latest_model_key"] = latest_label.lower().replace("-", "_")
        results["_latest_phase11_file"] = latest_file

    # Phase 14 ablation results
    phase14_results = {}
    for f in glob.glob(os.path.join(EXPERIMENTS, "phase14_final", "*_ablation_results.json")):
        model_name = os.path.basename(f).replace("_ablation_results.json", "")
        phase14_results[model_name] = _load(f)
    results["phase14_results"] = phase14_results

    # Phase 14 sensitivity maps (oracle cost, etc)
    phase14_maps = {}
    for f in glob.glob(os.path.join(EXPERIMENTS, "phase14_final", "*_sensitivity_map.json")):
        model_name = os.path.basename(f).replace("_sensitivity_map.json", "")
        phase14_maps[model_name] = _load(f)
    results["phase14_maps"] = phase14_maps

    # Phase 15 layer drop results
    phase15_results = {}
    for f in glob.glob(os.path.join(EXPERIMENTS, "phase15_layer_drop", "*.json")):
        key = os.path.basename(f).replace(".json", "")
        phase15_results[key] = _load(f)
    results["phase15_results"] = phase15_results

    # Count total experiments
    all_json = [
        path for path in glob.glob(os.path.join(EXPERIMENTS, "**", "*.json"), recursive=True)
        if not any("_pruned_" in part for part in Path(path).parts)
    ]
    results["_total_experiments"] = len(all_json)

    return results


def extract_metrics(data):
    """Extract deployment metrics from phase 11 results."""
    if not data:
        return None
    try:
        def pct_fraction(value):
            if value is None:
                return 0
            return value / 100.0 if abs(value) > 1 else value

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
        deploy = s4.get("deployment_metrics", {}) or s4.get("deployment", {})
        baseline = s4.get("baseline", {})
        pruned = s4.get("pruned_before_healing", {})
        healed = s4.get("healed", {})

        baseline_params_b = deploy.get("baseline_parameters_b")
        if baseline_params_b is None and baseline.get("params") is not None:
            baseline_params_b = baseline["params"] / 1e9

        healed_params_b = deploy.get("healed_parameters_b")
        if healed_params_b is None and pruned.get("nonzero_params") is not None:
            healed_params_b = pruned["nonzero_params"] / 1e9

        return {
            "winning_sparsity": winner.get("actual_sparsity", winner.get("sparsity", winner.get("target_sparsity", 0))),
            "ppl_loss_pct": pct_fraction(
                winner.get("ppl_loss_increase_pct", winner.get("ppl_increase_pct", 0))
            ),
            "Phase 11: Real Measurement": "done",
            "Phase 12: Recovery Diagnostics": "done",
            "Phase 13: Ablation & Hardening": "done",
            "critical_layers": critical[:5],
            "safe_layers": safe[:5],
            "covariance_restoration": s4.get("covariance_restoration_score", 0),
            "layers_healed": s4.get("layers_healed_count", s4.get("layers_bias_corrected", 0)),
            "baseline_params_b": baseline_params_b or 0,
            "healed_params_b": healed_params_b or 0,
            "baseline_latency_ms": deploy.get("baseline_latency_ms", baseline.get("latency_ms", 0)),
            "healed_latency_ms": deploy.get("healed_latency_ms", healed.get("latency_ms", 0)),
            "baseline_memory_gb": deploy.get("baseline_memory_gb", baseline.get("memory_gb", 0)),
            "healed_memory_gb": deploy.get("healed_memory_gb", healed.get("memory_gb", 0)),
            "perplexity_recovery_pct": deploy.get("perplexity_recovery_pct", healed.get("ppl_recovery_pct", 0)),
        }
    except Exception:
        return None


def build_state_json(experiments, metrics):
    """Build the machine-readable state.json for agent consumption."""
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        "project": "TIM-MGDT",
        "tagline": TAGLINE,
        "current_phase": CURRENT_PHASE,
        "phase_status": PHASE,
        "total_experiments": experiments.get("_total_experiments", 0),
        "latest_run": {
            "name": f"{experiments.get('_latest_model_name', 'Mistral-7B')} Phase 11 (Real GPU)",
            "status": "complete" if experiments.get("phase11") else "pending",
            "model": experiments.get("_latest_model_key", "mistral_7b"),
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


def _phase14_html(results, maps):
    import math
    # We want to display results for Mistral-7B-v0.1 and Meta-Llama-3-8B.
    models = [
        ("Mistral-7B-v0.1", "Mistral 7B (MHA)"),
        ("Meta-Llama-3-8B", "Llama 3 8B (GQA)")
    ]
    
    html = '<div class="section-label" id="phase14">Legacy Explorations — Phase 14 Surgical Recovery</div>'
    html += '<details style="margin-bottom: 24px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 16px;">'
    html += '<summary style="cursor: pointer; font-weight: 600; color: var(--ink); outline: none;">View Phase 14 Surgical (ΔCE) Results</summary>'
    html += '<div style="margin-top: 16px;">'
    
    for key, display_name in models:
        model_results = results.get(key, {})
        model_map = maps.get(key, {})
        
        oracle_cost = "Pending..."
        if model_map and "oracle_compute_time_min" in model_map:
            oracle_cost = f"<b>{model_map['oracle_compute_time_min']:.2f} minutes</b>"
            
        # Get sensitivities to list surgical targeted layers
        sens = model_map.get("sensitivities", {})
        if sens:
            sorted_layers = sorted(sens.items(), key=lambda x: x[1], reverse=True)
            top_6 = [int(layer.split("_")[1]) for layer, score in sorted_layers[:6]]
            surgical_layers_str = str(sorted(top_6))
        else:
            surgical_layers_str = "Pending sensitivity map..."

        # Helper to extract metric row
        def get_row(mode_name, layers_targeted, run_key):
            run = model_results.get(run_key)
            if not run:
                return f"""<tr><td class="task-who">{mode_name}</td><td><code>{layers_targeted}</code></td><td class="task-action">—</td><td class="task-action">—</td><td class="task-action">—</td><td class="task-action">—</td></tr>"""
            
            baseline = run.get("baseline_c4_ppl")
            pruned = run.get("pruned_c4_ppl")
            recovered = run.get("recovered_c4_ppl")
            delta = run.get("c4_delta_pct")
            time_s = run.get("recovery_time_s")
            
            pruned_str = f"{pruned:.3f}" if pruned is not None else "—"
            recovered_str = f"{recovered:.3f}" if recovered is not None else "—"
            
            if delta is not None:
                delta_str = f"{delta:+.2f}%"
                if delta > 15.0:
                    delta_str = f"<span style='color:var(--red); font-weight:600;'>{delta_str}</span>"
                elif delta < 5.0:
                    delta_str = f"<span style='color:var(--green); font-weight:600;'>{delta_str}</span>"
                else:
                    delta_str = f"<span style='color:var(--yellow); font-weight:600;'>{delta_str}</span>"
            else:
                delta_str = "—"
                
            time_str = f"{time_s/60:.1f} min" if time_s is not None else "—"
            
            return f"""
            <tr>
              <td class="task-who">{mode_name}</td>
              <td><code>{layers_targeted}</code></td>
              <td class="task-action">{pruned_str}</td>
              <td class="task-action"><b>{recovered_str}</b></td>
              <td class="task-action">{delta_str}</td>
              <td class="task-action">{time_str}</td>
            </tr>
            """
        
        # Dense baseline row
        dense_ppl = "—"
        if model_map and "baseline_loss" in model_map:
            try:
                dense_ppl = f"{math.exp(model_map['baseline_loss']):.3f}"
            except Exception:
                pass
        else:
            for r_key in ["uniform_20pct", "ends_only_20pct", "surgical_20pct"]:
                if r_key in model_results:
                    val = model_results[r_key].get("baseline_c4_ppl")
                    if val is not None:
                        dense_ppl = f"{val:.3f}"
                        break
        
        dense_row = f"""
        <tr>
          <td class="task-who">Dense Baseline</td>
          <td>N/A (All Layers Intact)</td>
          <td class="task-action">—</td>
          <td class="task-action"><b>{dense_ppl}</b></td>
          <td class="task-action">0.00%</td>
          <td class="task-action">—</td>
        </tr>
        """
        
        uniform_row = get_row("Uniform Random (All Layers)", "All 32 Layers", "uniform_20pct")
        ends_only_row = get_row("Ends-Only Heuristic", "[0, 1, 2, 29, 30, 31]", "ends_only_20pct")
        surgical_row = get_row("Surgical Random (ΔCE)", surgical_layers_str, "surgical_20pct")
        
        html += f"""
        <div class="card" style="margin-bottom: 24px; padding: 0; overflow: hidden;">
          <div class="rc-head" style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <h3>{display_name}</h3>
              <div class="rc-meta">Live C4 Held-Out Perplexity recovery validation</div>
            </div>
            <div class="nav-meta" style="text-align:right;">
              Oracle Compute: {oracle_cost}
            </div>
          </div>
          <table class="task-table" style="width:100%;">
            <thead>
              <tr>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Allocation Method</th>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Layers Targeted</th>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Sparse C4 PPL (20%)</th>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Recovered C4 PPL</th>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Δ Recovery vs Dense</th>
                <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Time (Wall-Clock)</th>
              </tr>
            </thead>
            <tbody>
              {dense_row}
              {uniform_row}
              {ends_only_row}
              {surgical_row}
            </tbody>
          </table>
        </div>
        """
    html += '</div></details>'
    return html


def _phase15_html(phase15_results):
    models = [
        ("Mistral-7B-v0.1", "Mistral 7B (MHA)"),
        ("Meta-Llama-3-8B", "Llama 3 8B (GQA)"),
        ("gemma-2-2b", "Gemma 2 2B (SWA)")
    ]
    
    html = '<div class="section-label" id="phase15">Phase 15: Incision Boundary Healing for Depth-Pruned LLMs</div>'
    
    html += """
    <div class="card" style="margin-bottom: 24px; padding: 18px;">
      <p style="margin: 0 0 16px 0; color: var(--ink2); line-height: 1.6;">
        While recent literature (e.g., <i>Gromov et al., ShortGPT, LaCo</i>) has established that <b>Layer Dropping</b> achieves true hardware compression, the standard practice of applying global fine-tuning to all surviving layers to recover performance is highly sub-optimal.
      </p>
      <p style="margin: 0 0 16px 0; color: var(--ink2); line-height: 1.6;">
        <b>Our Primary Contribution — Incision Boundary Healing:</b> We empirically prove that dropping contiguous layers creates a sharp, highly localized discontinuity in the activation flow, rather than global degradation. Our <b>Boundary-Focused Surgical LoRA</b> restricts trainable parameters exclusively to the layers adjacent to the "incision". This routes the compute budget with precision, recovering up to 4x more perplexity than global baselines while using 33% fewer trainable parameters.
      </p>
      <table class="task-table" style="width:100%;">
        <thead>
          <tr>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:left; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Model</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Layers Dropped</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Latency Reduction</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Baseline C4 PPL</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Dropped C4 PPL</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Healed (Global) PPL</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Healed (Boundary) PPL</th>
          </tr>
        </thead>
        <tbody>
    """
    
    for key, display_name in models:
        eval_data = phase15_results.get(f"{key}_drop4_eval", {})
        healed_data = phase15_results.get(f"{key}_drop4_healed", {})
        boundary_data = phase15_results.get(f"{key}_drop4_boundary_healed", {})
        
        drop_indices = eval_data.get("drop_indices", healed_data.get("drop_indices", []))
        drop_indices_str = str(drop_indices) if drop_indices else "—"
        
        base_lat = eval_data.get("baseline_latency_ms", 0.0)
        drop_lat = eval_data.get("dropped_latency_ms", 0.0)
        lat_saved = eval_data.get("latency_reduction_pct", 0.0)
        
        lat_saved_str = f"-{lat_saved:.2f}% ({base_lat:.1f}ms → {drop_lat:.1f}ms)" if lat_saved else "—"
        
        base_ppl = eval_data.get("baseline_c4_ppl", healed_data.get("baseline_c4_ppl"))
        base_ppl_str = f"{base_ppl:.3f}" if base_ppl else "—"
        
        drop_ppl = eval_data.get("dropped_c4_ppl", healed_data.get("dropped_c4_ppl"))
        drop_ppl_str = f"{drop_ppl:.3f}" if drop_ppl else "—"
        
        healed_ppl = healed_data.get("healed_c4_ppl")
        healed_ppl_str = f"{healed_ppl:.3f}" if healed_ppl else "—"
        
        boundary_ppl = boundary_data.get("healed_c4_ppl")
        boundary_ppl_str = f"{boundary_ppl:.3f}" if boundary_ppl else "<i>Pending...</i>"
        
        html += f"""
        <tr>
          <td class="task-who">{display_name}</td>
          <td style="text-align:center;"><code>{drop_indices_str}</code></td>
          <td style="text-align:center; color:var(--green); font-weight:600;">{lat_saved_str}</td>
          <td style="text-align:center;">{base_ppl_str}</td>
          <td style="text-align:center; color:var(--red);">{drop_ppl_str}</td>
          <td style="text-align:center;">{healed_ppl_str}</td>
          <td style="text-align:center; font-weight:600; color:var(--blue);">{boundary_ppl_str}</td>
        </tr>
        """
        
    html += """
        </tbody>
      </table>
    </div>
    """
    
    # Phase 16B Pareto Frontier
    html += '<div class="section-label" id="pareto">Phase 16B: The Latency-Perplexity Pareto Frontier</div>'
    html += """
    <div class="card" style="margin-bottom: 24px; padding: 18px;">
      <p style="margin: 0 0 16px 0; color: var(--ink2); line-height: 1.6;">
        To build a defensible submission for COLM, we swept across multiple compression ratios (n_drop = 2, 4, 6, 8) on Mistral-7B. This establishes the Pareto frontier of true hardware latency reduction vs perplexity recovery using our Incision Boundary Healing. Note the massive relative recovery at 8 drops: boundary healing brings perplexity down from a completely broken 20.82 back to 11.52.
      </p>
      <div style="background-color: var(--surface2); border-left: 4px solid var(--primary); padding: 12px 16px; margin-bottom: 20px; border-radius: 0 4px 4px 0;">
        <h4 style="margin: 0 0 8px 0; color: var(--ink); font-size: 14px;">The Pareto "Knee" Caveat</h4>
        <p style="margin: 0; color: var(--ink2); font-size: 13px; line-height: 1.5;">
          While recovering 9 points of perplexity is an incredible relative win, an absolute perplexity of 11.517 means the 8-layer compressed model is still heavily degraded compared to the 7.855 baseline and will likely fail complex reasoning tasks. We formally designate the <b>4-layer drop (-15.62% latency at 9.073 PPL)</b> as the optimal operating point (the "knee" of the curve)—the most actionable intersection of physical compression and retained capability.
        </p>
      </div>
      <div style="text-align: center; margin-bottom: 24px;">
        <img src="figures/bathtub_curve.png" alt="The Bathtub Curve" style="max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 16px;" />
        <br/>
        <img src="figures/pareto_frontier.png" alt="Latency-Perplexity Pareto Frontier" style="max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 16px;" />
        <br/>
        <img src="figures/thickness_ablation.png" alt="Boundary Thickness Ablation" style="max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);" />
      </div>
      <div style="background-color: var(--surface2); border-left: 4px solid var(--primary); padding: 12px 16px; margin-bottom: 20px; border-radius: 0 4px 4px 0;">
        <h4 style="margin: 0 0 8px 0; color: var(--ink); font-size: 14px;">The "Razor-Thin" Semantic Gap (Ablation 16C)</h4>
        <p style="margin: 0; color: var(--ink2); font-size: 13px; line-height: 1.5;">
          Our thickness ablation proved that the semantic damage is highly localized. Healing just <b>2 layers</b> (t=1) recovered 90% of the possible performance (9.163 PPL), and expanding the boundary beyond 4 layers yielded zero additional recovery. This confirms the boundary discontinuity hypothesis mathematically.
        </p>
      </div>
      <table class="task-table" style="width:100%;">
        <thead>
          <tr>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Layers Dropped</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Latency Reduction</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Baseline PPL</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Dropped PPL</th>
            <th style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--ink3); text-align:center; padding:10px 18px; border-bottom:1px solid var(--border); background:var(--surface2);">Healed (Boundary) PPL</th>
          </tr>
        </thead>
        <tbody>
    """
    
    for drop in [2, 4, 6, 8]:
        eval_data = phase15_results.get(f"Mistral-7B-v0.1_drop{drop}_eval", {})
        boundary_data = phase15_results.get(f"Mistral-7B-v0.1_drop{drop}_boundary_healed", {})
        
        if not eval_data:
            continue
            
        lat_saved = eval_data.get("latency_reduction_pct", 0.0)
        base_ppl = eval_data.get("baseline_c4_ppl", 0.0)
        drop_ppl = eval_data.get("dropped_c4_ppl", 0.0)
        healed_ppl = boundary_data.get("healed_c4_ppl", 0.0)
        
        html += f"""
        <tr>
          <td style="text-align:center; font-weight:600; color:var(--ink);">{drop} Layers</td>
          <td style="text-align:center; color:var(--green); font-weight:600;">-{lat_saved:.2f}%</td>
          <td style="text-align:center;">{base_ppl:.3f}</td>
          <td style="text-align:center; color:var(--red);">{drop_ppl:.3f}</td>
          <td style="text-align:center; font-weight:600; color:var(--blue);">{healed_ppl:.3f}</td>
        </tr>
        """
        
    html += """
        </tbody>
      </table>
    </div>
    """
    
    # Phase 17 Mechanistic Interpretability
    html += '<div class="section-label" id="phase17">Phase 17: Mechanistic Interpretability</div>'
    html += """
    <div class="card" style="margin-bottom: 24px; padding: 18px;">
      <p style="margin: 0 0 16px 0; color: var(--ink2); line-height: 1.6;">
        To provide mechanistic evidence of the structural damage induced by the layer drop, we tracked the average attention mass directed toward the Beginning-Of-Sequence (BOS) sink token. LLM attention heads default to the BOS token when uncertain or when their incoming residual stream is disrupted.
      </p>
      <div style="background-color: var(--surface2); border-left: 4px solid var(--accent); padding: 12px 16px; margin-bottom: 20px; border-radius: 0 4px 4px 0;">
        <h4 style="margin: 0 0 8px 0; color: var(--ink); font-size: 14px;">Attention Sink Spike (Layer 18)</h4>
        <p style="margin: 0; color: var(--ink2); font-size: 13px; line-height: 1.5;">
          The massive spike in sink attention exactly at the right boundary layer (Layer 18) mathematically confirms that the structural discontinuity is severely corrupting the incoming residual stream. The model "notices" the incision, causing boundary layers to dump their attention mass into the sink.
        </p>
      </div>
      <div style="text-align: center;">
        <img src="figures/attention_heatmap.png" alt="Mechanistic Disruption: Attention Mass Directed to Sink Token" style="max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);" />
      </div>
    </div>
    """
    
    return html


def _gsm8k_metric(path):
    data = _load(path)
    if not data:
        return None
    gsm8k = data.get("results", {}).get("gsm8k", {})
    samples = data.get("n-samples", {}).get("gsm8k", {})
    config = data.get("config", {})
    return {
        "n": samples.get("effective", gsm8k.get("sample_len")),
        "strict": gsm8k.get("exact_match,strict-match"),
        "flex": gsm8k.get("exact_match,flexible-extract"),
        "params": config.get("model_num_parameters"),
        "limit": config.get("limit"),
    }


def _pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _params(value):
    if value is None:
        return "—"
    return f"{value / 1e9:.2f}B"


def _float(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _latest_gsm8k(pattern):
    def score(path):
        metric = _gsm8k_metric(path) or {}
        n = metric.get("n") or 0
        return (n, os.path.getmtime(path))

    paths = sorted(glob.glob(pattern), key=score)
    if not paths:
        return None
    return _gsm8k_metric(paths[-1])


def _phase18b_readout(sparsity, strict, dense_strict):
    if sparsity == 0:
        return "50-sample dense smoke baseline for the boundary sweep."
    if strict is None or dense_strict is None:
        return "Awaiting paired GSM8K smoke metric."

    drop = dense_strict - strict
    if sparsity <= 0.01 and drop >= 0.10:
        return "Even 1% structural pruning loses reasoning accuracy; do not scale uniform width pruning."
    if sparsity <= 0.025 and drop >= 0.10:
        return "Boundary check confirms the damage starts below 2.5%."
    if strict <= 0.10:
        return "Severe reasoning collapse."
    return "Reasoning degradation increases with sparsity."


def _phase18b_sweep_html():
    pruning_paths = sorted(
        glob.glob("experiments/phase18b/Mistral-7B-v0.1_width_pruning_ceil_s*.json"),
        key=lambda p: (_load(p) or {}).get("sparsity", 1),
    )
    if not pruning_paths:
        return ""

    baseline_metric = _latest_gsm8k("experiments/phase18b/gsm8k_eval_baseline*.json") or {}
    first_pruning = _load(pruning_paths[0]) or {}
    baseline_c4 = first_pruning.get("baseline_c4_ppl")
    baseline_tps = first_pruning.get("baseline_tps")
    dense_strict = baseline_metric.get("strict")

    rows = f"""
        <tr>
          <td class="task-who">Dense smoke baseline</td>
          <td>0.0%</td>
          <td>{int(baseline_metric.get("n", 0)) if baseline_metric.get("n") else "—"}</td>
          <td>{_pct(dense_strict)}</td>
          <td><span class="muted">ref</span></td>
          <td>{_float(baseline_c4)}</td>
          <td>{_float(baseline_tps, 2)}</td>
          <td class="task-action">{_phase18b_readout(0, dense_strict, dense_strict)}</td>
        </tr>
    """

    for path in pruning_paths:
        data = _load(path) or {}
        tag = Path(path).stem.split("_")[-1]
        metric = _latest_gsm8k(f"experiments/phase18b/gsm8k_eval_ceil_{tag}*.json") or {}
        sparsity = data.get("sparsity")
        strict = metric.get("strict")
        delta = None if strict is None or dense_strict is None else strict - dense_strict
        delta_cls = "neg" if delta is not None and delta < -0.02 else "zero"
        delta_str = "—" if delta is None else f"<span class='b b-{delta_cls}'>{delta * 100:+.1f} pt</span>"
        c4 = data.get("pruned_c4_ppl")
        tps = data.get("pruned_tps")
        n = metric.get("n")
        sparsity_str = f"{sparsity * 100:.1f}%" if sparsity is not None else "—"
        rows += f"""
        <tr>
          <td class="task-who">Phase 18B uniform structural ({tag})</td>
          <td>{sparsity_str}</td>
          <td>{int(n) if n else "—"}</td>
          <td>{_pct(strict)}</td>
          <td>{delta_str}</td>
          <td>{_float(c4)}</td>
          <td>{_float(tps, 2)}</td>
          <td class="task-action">{_phase18b_readout(sparsity, strict, dense_strict)}</td>
        </tr>
        """

    return f"""
<div class="section-label">Phase 18B boundary sweep</div>
<div class="card" style="padding:0;overflow:hidden;margin-bottom:18px">
<table class="task-table">
<tr><th>Probe</th><th>Sparsity</th><th>GSM8K n</th><th>Strict EM</th><th>Δ vs dense</th><th>C4 PPL</th><th>Tok/s</th><th>Readout</th></tr>
{rows}
</table>
<div class="rc-note">Smoke metrics use 5-shot GSM8K with <code>--limit 50</code>. They are for ranking directions before expensive full evaluations.</div>
</div>"""


def _phase19_mask_readout(tag, strict, c4, dense_strict=None):
    if strict is None:
        return "Awaiting GSM8K metric."
    if dense_strict is not None and tag == "s20":
        drop_pts = (strict - dense_strict) * 100
        if drop_pts >= -2.1:
            return "Result is within about 2 points of dense; strongest positive control."
        return "Best mask point so far, but full validation shows a measurable dense gap."
    if tag == "s20":
        return "Near-dense GSM8K at 20% mask sparsity; strongest positive control."
    if tag == "s30":
        return "C4-matched to structural 5%, but retains more GSM8K reasoning."
    if tag == "s40":
        return "Mask boundary: reasoning starts to collapse once C4 also degrades."
    if c4 is not None and c4 < 8.4 and strict >= 0.38:
        return "Promising mask point; candidate for full GSM8K validation."
    return "Mask pruning control point."


def _phase19_mask_html():
    mask_paths = sorted(
        glob.glob("experiments/phase19/Mistral-7B-v0.1_mask_pruning_s*.json"),
        key=lambda p: (_load(p) or {}).get("target_sparsity", 1),
    )
    if not mask_paths:
        return ""

    dense_full_metric = (
        _latest_gsm8k("experiments/phase19/gsm8k_eval_baseline_full*.json")
        or _gsm8k_metric("experiments/phase16/gsm8k_eval_baseline_2026-06-23T01-43-56.210011.json")
        or {}
    )
    dense_smoke_metric = _latest_gsm8k("experiments/phase18b/gsm8k_eval_baseline*.json") or {}
    dense_metric = (
        dense_full_metric
        if dense_full_metric.get("n")
        else dense_smoke_metric
        or _latest_gsm8k("experiments/phase18b/gsm8k_eval_baseline*.json")
        or {}
    )
    dense_strict = dense_metric.get("strict")
    dense_n = dense_metric.get("n")
    dense_badge = "full" if isinstance(dense_n, (int, float)) and dense_n > 100 else "smoke"
    first_mask = _load(mask_paths[0]) or {}
    baseline_c4 = first_mask.get("baseline_c4_ppl")
    baseline_tps = first_mask.get("baseline_tps")
    sample_analysis = _load("experiments/phase19/gsm8k_sample_analysis.json") or {}
    sample_by_tag = {row.get("tag"): row for row in sample_analysis.get("rows", [])}
    dense_sample = sample_by_tag.get("dense") or {}
    s20_sample = sample_by_tag.get("s20") or {}
    sample_note = ""
    if (
        dense_sample.get("avg_response_words") is not None
        and s20_sample.get("avg_response_words") is not None
        and s20_sample.get("delta_avg_words_vs_dense_full") is not None
    ):
        word_delta = s20_sample.get("delta_avg_words_vs_dense_full")
        marker_delta = None
        if dense_sample.get("final_marker_rate") is not None and s20_sample.get("final_marker_rate") is not None:
            marker_delta = (s20_sample["final_marker_rate"] - dense_sample["final_marker_rate"]) * 100
        sample_note = (
            "Sample analysis: the 20% mask keeps generated GSM8K chain length essentially unchanged "
            f"({s20_sample.get('avg_response_words'):.2f} vs {dense_sample.get('avg_response_words'):.2f} avg words, "
            f"{word_delta:+.2f} delta)"
        )
        if marker_delta is not None:
            sample_note += f" with a {marker_delta:+.1f} pt final-answer-marker change."
        sample_note = f"<div class=\"rc-note\">{sample_note}</div>"
    rows = f"""
        <tr>
          <td class="task-who">Dense Mistral baseline</td>
          <td>0.0%</td>
          <td>0.0%</td>
          <td>{int(dense_n) if dense_n else "—"} <span class="muted">({dense_badge})</span></td>
          <td>{_pct(dense_strict)}</td>
          <td><span class="muted">ref</span></td>
          <td>{_float(baseline_c4)}</td>
          <td>{_float(baseline_tps, 2)}</td>
          <td class="task-action">Reference for Phase 19 mask-vs-structural controls.</td>
        </tr>
"""
    for path in mask_paths:
        data = _load(path) or {}
        tag = Path(path).stem.split("_")[-1]
        metric = _latest_gsm8k(f"experiments/phase19/gsm8k_eval_mask_{tag}*.json") or {}
        target = data.get("target_sparsity")
        actual = data.get("linear_sparsity")
        strict = metric.get("strict")
        row_dense_metric = dense_full_metric if isinstance(metric.get("n"), (int, float)) and metric.get("n") > 100 else dense_smoke_metric
        row_dense_strict = row_dense_metric.get("strict", dense_strict)
        delta = None if strict is None or row_dense_strict is None else strict - row_dense_strict
        c4 = data.get("masked_c4_ppl")
        n = metric.get("n")
        tps = data.get("masked_tps")
        target_str = f"{target * 100:.1f}%" if target is not None else "—"
        actual_str = f"{actual * 100:.1f}%" if actual is not None else "—"
        full_badge = "full" if isinstance(n, (int, float)) and n > 100 else "smoke"
        delta_cls = "pos" if delta is not None and delta > 0.02 else ("neg" if delta is not None and delta < -0.02 else "zero")
        delta_str = "—" if delta is None else f"<span class='b b-{delta_cls}'>{delta * 100:+.1f} pt</span>"
        rows += f"""
        <tr>
          <td class="task-who">Phase 19 mask ({tag})</td>
          <td>{target_str}</td>
          <td>{actual_str}</td>
          <td>{int(n) if n else "—"} <span class="muted">({full_badge})</span></td>
          <td>{_pct(strict)}</td>
          <td>{delta_str}</td>
          <td>{_float(c4)}</td>
          <td>{_float(tps, 2)}</td>
          <td class="task-action">{_phase19_mask_readout(tag, strict, c4, row_dense_strict)}</td>
        </tr>
        """

    return f"""
<div class="section-label">Phase 19 mask control</div>
<div class="card" style="padding:0;overflow:hidden;margin-bottom:18px">
<table class="task-table">
<tr><th>Probe</th><th>Target</th><th>Actual Linear Sparsity</th><th>GSM8K n</th><th>Strict EM</th><th>Δ vs dense</th><th>C4 PPL</th><th>Tok/s</th><th>Readout</th></tr>
{rows}
</table>
<div class="rc-note">Dense-shaped unstructured masks do not provide structural speedups. Deltas compare against dense runs at the same sample scale when available, isolating whether reasoning loss comes from sparsity itself or from structural MLP-width geometry.</div>
{sample_note}
</div>
<div class="card" style="padding:18px;margin-bottom:18px">
  <img src="figures/phase19_reasoning_dissociation.png" alt="Phase 19 mask-vs-structural C4 and GSM8K dissociation plot" style="display:block;width:100%;height:auto;border:1px solid var(--border);border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.05)" />
</div>"""


def _phase20_mask_guided_html():
    result_paths = sorted(
        glob.glob("experiments/phase20_mask_guided/Mistral-7B-v0.1_mask_guided_width_*.json"),
        key=lambda p: (_load(p) or {}).get("target_structured_sparsity", 1),
    )
    if not result_paths:
        return ""

    dense_smoke = _latest_gsm8k("experiments/phase18b/gsm8k_eval_baseline*.json") or {}
    dense_strict = dense_smoke.get("strict")
    rows = ""
    for path in result_paths:
        data = _load(path) or {}
        tag = Path(path).stem.split("_")[-1]
        metric = _latest_gsm8k(f"experiments/phase20_mask_guided/gsm8k_eval_mask_guided_*_{tag}*.json") or {}
        strict = metric.get("strict")
        delta = None if strict is None or dense_strict is None else strict - dense_strict
        delta_cls = "pos" if delta is not None and delta > 0.02 else ("neg" if delta is not None and delta < -0.02 else "zero")
        delta_str = "—" if delta is None else f"<span class='b b-{delta_cls}'>{delta * 100:+.1f} pt</span>"
        readout = (
            "Fails continuation gate; do not scale larger mask-guided structured sparsities."
            if strict is not None and strict < 0.30
            else "Awaiting smoke metric."
            if strict is None
            else "Promising enough for follow-up."
        )
        rows += f"""
        <tr>
          <td class="task-who">Phase 20 mask-guided ({tag})</td>
          <td>{_pct(data.get("target_structured_sparsity"))}</td>
          <td>{_pct(data.get("mlp_param_sparsity"))}</td>
          <td>{int(metric.get("n")) if metric.get("n") else "—"}</td>
          <td>{_pct(strict)}</td>
          <td>{delta_str}</td>
          <td>{_float(data.get("pruned_c4_ppl"))}</td>
          <td>{_float(data.get("pruned_tps"), 2)}</td>
          <td class="task-action">{readout}</td>
        </tr>
        """

    return f"""
<div class="section-label">Phase 20 mask-guided structural transfer</div>
<div class="card" style="padding:0;overflow:hidden;margin-bottom:18px">
<table class="task-table">
<tr><th>Probe</th><th>Target</th><th>Actual MLP Sparsity</th><th>GSM8K n</th><th>Strict EM</th><th>Δ vs dense smoke</th><th>C4 PPL</th><th>Tok/s</th><th>Readout</th></tr>
{rows}
</table>
<div class="rc-note">Phase 20 tests whether the successful dense-mask signal transfers to real MLP-width deletion. Initial smoke results are negative, so larger points are stopped by default.</div>
</div>"""


def _phase23_global_readout(strict, dense_strict):
    if strict is None:
        return "Full GSM8K is running; this closes the standard global magnitude baseline gap."
    if dense_strict is not None and (strict - dense_strict) * 100 >= -2.1:
        return "Near-dense full GSM8K; supports unstructured geometry preservation over structural deletion."
    if dense_strict is not None and strict < 0.30:
        return "Global magnitude collapses; layerwise allocation is materially stronger."
    return "Global magnitude is weaker than the 20% layerwise mask but still far above structural width pruning."


def _phase23_global_html():
    meta = _load("experiments/phase23_global_unstructured/Mistral-7B-v0.1_mask_pruning_global_s20.json") or {}
    metric = _latest_gsm8k("experiments/phase23_global_unstructured/gsm8k_eval_global_s20*.json") or {}
    if not meta and not metric:
        return ""

    dense_metric = _latest_gsm8k("experiments/phase19/gsm8k_eval_baseline_full*.json") or {}
    dense_strict = dense_metric.get("strict")
    strict = metric.get("strict")
    delta = None if strict is None or dense_strict is None else strict - dense_strict
    delta_cls = "pos" if delta is not None and delta > 0.02 else ("neg" if delta is not None and delta < -0.02 else "zero")
    delta_str = "—" if delta is None else f"<span class='b b-{delta_cls}'>{delta * 100:+.1f} pt</span>"
    n = metric.get("n")

    rows = f"""
        <tr>
          <td class="task-who">Phase 23 global magnitude (s20)</td>
          <td>{_pct(meta.get("target_sparsity"))}</td>
          <td>{_pct(meta.get("linear_sparsity"))}</td>
          <td>{int(n) if n else "—"}</td>
          <td>{_pct(strict)}</td>
          <td>{delta_str}</td>
          <td>{_float(meta.get("masked_c4_ppl"))}</td>
          <td>{_float(meta.get("masked_tps"), 2)}</td>
          <td class="task-action">{_phase23_global_readout(strict, dense_strict)}</td>
        </tr>
"""

    return f"""
<div class="section-label">Phase 23 global unstructured baseline</div>
<div class="card" style="padding:0;overflow:hidden;margin-bottom:18px">
<table class="task-table">
<tr><th>Probe</th><th>Target</th><th>Actual Linear Sparsity</th><th>GSM8K n</th><th>Strict EM</th><th>Delta vs dense</th><th>C4 PPL</th><th>Tok/s</th><th>Readout</th></tr>
{rows}
</table>
<div class="rc-note">This baseline answers whether the 20% retention point is a layerwise dense-mask artifact or also holds for standard global unstructured magnitude masking.</div>
</div>"""


def _width_pruning_html():
    phase16_ceil = _load("experiments/phase16/Mistral-7B-v0.1_width_pruning_ceil.json") or {}
    phase16_floor = _load("experiments/phase16/Mistral-7B-v0.1_width_pruning_floor.json") or {}
    rows = [
        (
            "Dense Mistral baseline",
            _gsm8k_metric("experiments/phase16/gsm8k_eval_baseline_2026-06-23T01-43-56.210011.json"),
            phase16_ceil.get("baseline_c4_ppl"),
            "Reference run, 5-shot GSM8K full eval.",
        ),
        (
            "Phase 16 width prune (ceil)",
            _gsm8k_metric("experiments/phase16/gsm8k_eval_ceil_2026-06-23T03-09-22.319762.json"),
            phase16_ceil.get("pruned_c4_ppl"),
            "Real structural compression; C4 acceptable, reasoning collapsed.",
        ),
        (
            "Phase 16 width prune (floor)",
            _gsm8k_metric("experiments/phase16/gsm8k_eval_floor_2026-06-23T01-51-52.825718.json"),
            phase16_floor.get("pruned_c4_ppl"),
            "Slightly smaller checkpoint; same reasoning failure mode.",
        ),
        (
            "Phase 17 contrastive sparing",
            _gsm8k_metric("experiments/phase17/gsm8k_eval_contrastive_ceil_2026-06-23T03-31-48.393338.json"),
            None,
            "50-sample smoke test failed; do not scale as-is.",
        ),
        (
            "Phase 18 non-uniform mask",
            _gsm8k_metric("experiments/phase18/gsm8k_eval_non_uniform_ceil_2026-06-23T03-56-46.992242.json"),
            None,
            "Diagnostic mask only; checkpoint keeps dense parameter shape.",
        ),
    ]

    body = ""
    for label, metric, c4_ppl, note in rows:
        metric = metric or {}
        n = metric.get("n")
        n_str = f"{int(n)}" if isinstance(n, (int, float)) else "—"
        c4_str = f"{c4_ppl:.3f}" if c4_ppl is not None else "—"
        body += f"""
        <tr>
          <td class="task-who">{label}</td>
          <td>{n_str}</td>
          <td>{_params(metric.get("params"))}</td>
          <td>{_pct(metric.get("strict"))}</td>
          <td>{c4_str}</td>
          <td class="task-action">{note}</td>
        </tr>
        """

    return f"""
<div class="section-label" id="current">Current pruning diagnostics</div>
{_phase18b_sweep_html()}
{_phase19_mask_html()}
{_phase23_global_html()}
{_phase20_mask_guided_html()}
<div class="card" style="padding:0;overflow:hidden">
<table class="task-table">
<tr><th>Probe</th><th>GSM8K n</th><th>Params</th><th>Strict EM</th><th>C4 PPL</th><th>Readout</th></tr>
{body}
</table>
</div>"""


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


def _metrics_html(metrics, model_name="Mistral-7B"):
    if not metrics:
        return "<div class='card'><p class='muted'>No metrics available yet.</p></div>"

    def pct_change(base, opt):
        if base == 0:
            return "—"
        return f"{100*(opt/base - 1):+.1f}%"

    return f"""
<div class="section-label">Historical Results — Phase 11 ({model_name})</div>
<details style="margin-bottom: 24px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 16px;">
<summary style="cursor: pointer; font-weight: 600; color: var(--ink); outline: none;">View Phase 11 Real GPU Baseline Metrics</summary>
<div class="g2" style="margin-top: 16px;">
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
    <div class="metric-cell"><span class="metric-label">Param Change</span>
    <span class="metric-value good">{pct_change(metrics['baseline_params_b'], metrics['healed_params_b'])}</span></div>
    <div class="metric-cell"><span class="metric-label">Latency Change</span>
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
</details>"""


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
    done_count = sum(1 for _, s in RUNS if s.startswith("done"))

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
    <a href="#current">Current Probe</a>
    <a href="#decisions">Decisions</a>
    <a href="#tasks">Tasks</a>
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
      <span class="chip chip-blue">{CURRENT_PHASE}</span>
      <span class="chip chip-green">{done_count}/{len(RUNS)} runs complete</span>
      <span class="chip chip-yellow">{total} experiment files</span>
      <span class="chip chip-gray">Llama-3 · Mistral-7B</span>
    </div>
  </div>
</div>

<div class="wrap">
  <div class="section-label" id="overview">Project overview</div>
  <div class="g3">
    {_overview_cards()}
  </div>
  <div style="margin-top:16px">{_research_callout()}</div>
  {_width_pruning_html()}

  <div style="margin-top: 32px;">
    {_phase15_html(experiments.get("phase15_results", {}))}
  </div>
  <div style="margin-top: 32px;">
    {_phase14_html(experiments.get("phase14_results", {}), experiments.get("phase14_maps", {}))}
  </div>
  <div id="results" style="margin-top: 32px;">{_metrics_html(metrics, experiments.get('_latest_model_name', 'Mistral-7B'))}</div>
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
