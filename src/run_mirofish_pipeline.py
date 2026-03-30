"""
Automated MiroFish pipeline for Calgary Housing simulation.
Hits the MiroFish Flask API end-to-end:
  1. Upload seed doc → generate ontology  (project_id)
  2. Build Zep knowledge graph            (graph_id)
  3. Create simulation                    (simulation_id)
  4. Prepare agents (LLM profiles)
  5. Start simulation                     (max_rounds configurable)
  6. Generate report
  7. Save report to simulations/
"""

import requests
import time
import os
import sys
import json

BASE_URL = "http://localhost:5001"

_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_date_str  = __import__("datetime").date.today().strftime("%Y-%m-%d")

_mirofish_base = os.environ.get(
    "MIROFISH_DIR",
    os.path.join(_repo_dir, "..", "MiroFish"),
)

_sim_subdir = f"{_date_str}-calgary-housing"
OUTPUT_DIR  = os.path.join(_mirofish_base, "simulations", _sim_subdir)

# Seed = latest generated report, or the pre-placed seed.md if it exists
_reports_dir = os.path.join(_repo_dir, "reports")
_existing_seed = os.path.join(OUTPUT_DIR, "seed.md")
if os.path.exists(_existing_seed):
    SEED_FILE = _existing_seed
else:
    _reports = sorted(
        [f for f in os.listdir(_reports_dir) if f.endswith(".md")],
        reverse=True
    ) if os.path.isdir(_reports_dir) else []
    SEED_FILE = os.path.join(_reports_dir, _reports[0]) if _reports else _existing_seed

SIMULATION_REQUIREMENT = (
    "Predict Calgary housing market outcomes for December 2026. "
    "Simulate key actors: Bank of Canada (rate policy), Alberta energy sector "
    "(employment/wages), Calgary home buyers (first-time/move-up/investor), "
    "Federal government (immigration/housing policy), global oil markets "
    "(OPEC+/US shale), Iran/US geopolitical actors (Strait of Hormuz). "
    "Key prediction targets: (1) Calgary Total Residential benchmark price Dec 2026, "
    "(2) Bank of Canada overnight rate Dec 2026, (3) WTI oil avg Q3/Q4 2026, "
    "(4) Calgary buyer vs seller market sentiment, (5) most at-risk property segment."
)

MAX_ROUNDS = 40  # budget-conscious; increase for deeper simulation


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def poll(url, method="GET", body=None, label="", interval=10, timeout=900):
    """Poll an endpoint until success/failure, return final response data."""
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            if method == "POST":
                r = requests.post(url, json=body, timeout=30)
            else:
                r = requests.get(url, timeout=30)
            data = r.json()
        except Exception as e:
            log(f"  [{label}] request error: {e}, retrying...")
            time.sleep(interval)
            continue

        status = (
            data.get("data", {}).get("status") or
            data.get("data", {}).get("runner_status") or
            data.get("status", "")
        )
        progress = data.get("data", {}).get("progress_percentage", "")
        progress_str = f" ({progress:.0f}%)" if isinstance(progress, (int, float)) else ""
        log(f"  [{label}] attempt {attempt}: status={status}{progress_str}")

        if status in ("completed", "ready", "stopped", "failed", "error"):
            return data
        # For run status: completed means simulation done
        if status == "completed":
            return data

        time.sleep(interval)

    raise TimeoutError(f"[{label}] timed out after {timeout}s")


def step1_generate_ontology():
    log("=== Step 1: Upload seed + generate ontology ===")
    seed_path = os.path.abspath(SEED_FILE)
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed file not found: {seed_path}")

    with open(seed_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/graph/ontology/generate",
            data={
                "simulation_requirement": SIMULATION_REQUIREMENT,
                "project_name": "Calgary Housing 2026",
            },
            files={"files": ("seed.md", f, "text/markdown")},
            timeout=120,
        )

    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Ontology generation failed: {data}")

    project_id = data["data"]["project_id"]
    entity_count = len(data["data"]["ontology"].get("entity_types", []))
    log(f"  project_id={project_id}, ontology: {entity_count} entity types")
    return project_id


def step2_build_graph(project_id):
    log("=== Step 2: Build Zep knowledge graph ===")
    r = requests.post(
        f"{BASE_URL}/api/graph/build",
        json={"project_id": project_id, "graph_name": "Calgary Housing 2026"},
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Graph build start failed: {data}")

    task_id = data["data"]["task_id"]
    log(f"  task_id={task_id}, polling...")

    result = poll(
        f"{BASE_URL}/api/graph/task/{task_id}",
        label="graph-build",
        interval=15,
        timeout=1200,
    )
    if result.get("data", {}).get("status") != "completed":
        raise RuntimeError(f"Graph build failed: {result}")

    # Get graph_id from project
    proj = requests.get(f"{BASE_URL}/api/graph/project/{project_id}", timeout=15).json()
    graph_id = proj["data"]["graph_id"]
    log(f"  graph_id={graph_id}")
    return graph_id


def step3_create_simulation(project_id, graph_id):
    log("=== Step 3: Create simulation ===")
    r = requests.post(
        f"{BASE_URL}/api/simulation/create",
        json={
            "project_id": project_id,
            "graph_id": graph_id,
            "enable_twitter": True,
            "enable_reddit": False,  # Twitter only to save quota
        },
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Simulation create failed: {data}")

    simulation_id = data["data"]["simulation_id"]
    log(f"  simulation_id={simulation_id}")
    return simulation_id


def step4_prepare_simulation(simulation_id):
    log("=== Step 4: Prepare agents (LLM profile generation) ===")
    r = requests.post(
        f"{BASE_URL}/api/simulation/prepare",
        json={"simulation_id": simulation_id},
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Prepare start failed: {data}")

    if data["data"].get("already_prepared"):
        log("  Already prepared, skipping.")
        return

    task_id = data["data"].get("task_id")
    log(f"  task_id={task_id}, polling...")

    result = poll(
        f"{BASE_URL}/api/simulation/prepare/status",
        method="POST",
        body={"task_id": task_id, "simulation_id": simulation_id},
        label="prepare",
        interval=15,
        timeout=900,
    )
    status = result.get("data", {}).get("status")
    if status not in ("ready", "completed"):
        raise RuntimeError(f"Prepare failed: {result}")
    log("  Agents prepared.")


def step5_run_simulation(simulation_id):
    log(f"=== Step 5: Run simulation (max_rounds={MAX_ROUNDS}) ===")
    r = requests.post(
        f"{BASE_URL}/api/simulation/start",
        json={
            "simulation_id": simulation_id,
            "platform": "twitter",
            "max_rounds": MAX_ROUNDS,
            "enable_graph_memory_update": False,
        },
        timeout=30,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Simulation start failed: {data}")

    log(f"  Started. Polling run status...")

    result = poll(
        f"{BASE_URL}/api/simulation/{simulation_id}/run-status",
        label="run",
        interval=20,
        timeout=3600,
    )
    runner_status = result.get("data", {}).get("runner_status")
    log(f"  Simulation finished with status: {runner_status}")
    return runner_status


def step6_generate_report(simulation_id):
    log("=== Step 6: Generate report ===")
    r = requests.post(
        f"{BASE_URL}/api/report/generate",
        json={"simulation_id": simulation_id},
        timeout=60,
    )
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Report generation start failed: {data}")

    report_id = data["data"]["report_id"]
    log(f"  report_id={report_id}, polling...")

    result = poll(
        f"{BASE_URL}/api/report/{report_id}/progress",
        label="report",
        interval=15,
        timeout=900,
    )
    log("  Report complete. Fetching content...")

    r2 = requests.get(f"{BASE_URL}/api/report/{report_id}", timeout=30)
    report_data = r2.json()
    content = report_data.get("data", {}).get("content", "")
    return report_id, content


def save_report(content, project_id, simulation_id, graph_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "report.md")

    header = f"""<!-- MiroFish Simulation Report
project_id: {project_id}
simulation_id: {simulation_id}
graph_id: {graph_id}
generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
-->

"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header + content)

    log(f"  Report saved to: {report_path}")
    log(f"  Length: {len(content):,} chars")
    return report_path


def main():
    log("Starting MiroFish Calgary Housing Pipeline")
    log(f"Seed: {os.path.abspath(SEED_FILE)}")
    log(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    log("")

    try:
        project_id   = step1_generate_ontology()
        graph_id     = step2_build_graph(project_id)
        simulation_id = step3_create_simulation(project_id, graph_id)
        step4_prepare_simulation(simulation_id)
        step5_run_simulation(simulation_id)
        report_id, content = step6_generate_report(simulation_id)
        report_path = save_report(content, project_id, simulation_id, graph_id)

        log("")
        log("=== Pipeline complete! ===")
        log(f"Report: {report_path}")
        log(f"Preview (first 500 chars):")
        print(content[:500])

    except Exception as e:
        log(f"PIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
