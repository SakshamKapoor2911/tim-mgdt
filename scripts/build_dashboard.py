import os
import json
import glob
from jinja2 import Environment, FileSystemLoader

EXPERIMENTS_DIR = "experiments"
DASHBOARD_DIR = "dashboard"

# Define static tasks for the board
TASKS = [
    {
        "name": "Analyze Phase 11 GPT-2 real results",
        "assigned_to": "Saksham",
        "status": "in_progress",
        "blocked_by": "None"
    },
    {
        "name": "Implement Llama3 Causal Patching",
        "assigned_to": "Edward",
        "status": "pending",
        "blocked_by": "Phase 11 GPT-2 Validation"
    },
    {
        "name": "Setup automated MCP server for state.json",
        "assigned_to": "Unassigned",
        "status": "pending",
        "blocked_by": "Dashboard Deployment"
    }
]

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None

def find_latest_experiment():
    # To prevent log file bloat, we only scan the specific latest phases or rely on naming conventions.
    # In this case, we'll scan phase11 and phase10.
    search_paths = [
        os.path.join(EXPERIMENTS_DIR, "phase11", "*.json"),
        os.path.join(EXPERIMENTS_DIR, "phase10", "*.json")
    ]
    
    all_json_files = []
    for path in search_paths:
        all_json_files.extend(glob.glob(path))
        
    total_experiments = len(all_json_files)
    
    latest_run = None
    latest_metrics = None
    latest_status = "failed"
    current_phase = "Phase 11"
    sparsity = 0.0

    # Look specifically for complete results from phase 11 as the latest
    target_file = os.path.join(EXPERIMENTS_DIR, "phase11", "gpt2_phase11_complete_results.json")
    if os.path.exists(target_file):
        data = load_json(target_file)
        if data:
            latest_run = "GPT-2 Phase 11"
            # Extract deployment metrics if available
            try:
                stage4 = data.get("stages", {}).get("stage4_activation_healing", {})
                latest_status = stage4.get("status", "failed")
                metrics = stage4.get("deployment_metrics", {})
                if metrics:
                    latest_metrics = metrics
                    # Get sparsity from stage2
                    stage2 = data.get("stages", {}).get("stage2_imp_search", {})
                    winner = stage2.get("best_winner", {})
                    sparsity = winner.get("sparsity", 0.0)
                    latest_metrics["sparsity"] = sparsity
            except Exception as e:
                print(f"Error parsing metrics from {target_file}: {e}")
                latest_status = "error"
    else:
        latest_status = "pending"
        latest_run = "Awaiting Phase 11 Results"

    return {
        "current_phase": current_phase,
        "total_experiments": total_experiments,
        "latest_run_name": latest_run,
        "latest_status": latest_status,
        "metrics": latest_metrics,
        "tasks": TASKS
    }

def main():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)
    
    # 1. Gather state
    state = find_latest_experiment()
    
    # 2. Output state.json for agent consumption
    state_path = os.path.join(DASHBOARD_DIR, "state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Wrote {state_path}")
    
    # 3. Output index.html for human consumption via Jinja2
    env = Environment(loader=FileSystemLoader(DASHBOARD_DIR))
    try:
        template = env.get_template("template.html")
        html_output = template.render(state=state)
        
        html_path = os.path.join(DASHBOARD_DIR, "index.html")
        with open(html_path, "w") as f:
            f.write(html_output)
        print(f"Wrote {html_path}")
    except Exception as e:
        print(f"Failed to render HTML template: {e}")

if __name__ == "__main__":
    main()
