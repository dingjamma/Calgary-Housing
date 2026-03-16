"""
Calgary Housing Intelligence Pipeline — full automated run.
Runs every 2 weeks via Windows Task Scheduler.

Steps:
  1. Scrape all data sources (oil, CREB, BoC, economic indicators)
  2. Merge datasets
  3. Retrain all three XGBoost models
  4. Scrape latest news
  5. Generate market intelligence report
  6. Start MiroFish backend (if not already running)
  7. Run MiroFish simulation pipeline
  8. Commit all outputs to GitHub
"""

import subprocess
import sys
import os
import time
import requests
import logging
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

REPO_DIR      = Path(__file__).parent
MIROFISH_DIR  = REPO_DIR.parent / "MiroFish"
PYTHON        = sys.executable
LOG_DIR       = REPO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MIROFISH_URL  = "http://localhost:5001"
MIROFISH_STARTUP_WAIT = 12   # seconds to wait after starting backend

# ── Logging ─────────────────────────────────────────────────────────────────

log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(script_name, cwd=None):
    """Run a Python script, raise on failure."""
    path = REPO_DIR / "src" / script_name
    log.info(f"Running: {script_name}")
    result = subprocess.run(
        [PYTHON, str(path)],
        cwd=str(cwd or REPO_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-5:]:  # last 5 lines
            log.info(f"  {line}")
    if result.returncode != 0:
        log.error(f"FAILED: {script_name}")
        log.error(result.stderr[-500:] if result.stderr else "(no stderr)")
        raise RuntimeError(f"{script_name} exited {result.returncode}")
    return result


def mirofish_is_up():
    try:
        r = requests.get(f"{MIROFISH_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def start_mirofish():
    if mirofish_is_up():
        log.info("MiroFish backend already running.")
        return None

    log.info("Starting MiroFish backend...")
    proc = subprocess.Popen(
        ["cmd", "/c", "npm run backend"],
        cwd=str(MIROFISH_DIR),
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    # Wait for it to come up
    for i in range(MIROFISH_STARTUP_WAIT * 2):
        time.sleep(0.5)
        if mirofish_is_up():
            log.info(f"MiroFish backend up after {(i+1)*0.5:.0f}s")
            return proc

    raise RuntimeError("MiroFish backend did not start within timeout")


def git_commit_and_push():
    log.info("Committing outputs to GitHub...")
    date_str = datetime.now().strftime("%Y-%m-%d")

    cmds = [
        ["git", "add", "data/raw/news_feed.csv", "data/raw/news_feed.md",
         "data/processed/", "reports/", "-f"],
        ["git", "commit", "-m",
         f"Automated pipeline run {date_str}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
         "--allow-empty"],
        ["git", "push", "origin", "master"],
    ]

    for cmd in cmds:
        result = subprocess.run(cmd, cwd=str(REPO_DIR), capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        log.info(f"  git {cmd[1]}: {result.returncode}")
        if result.returncode not in (0, 1):  # 1 = nothing to commit
            log.warning(result.stderr[:200])

    # Also push MiroFish simulation results
    sim_dir = MIROFISH_DIR / "simulations"
    mf_cmds = [
        ["git", "add", str(sim_dir)],
        ["git", "commit", "-m",
         f"Simulation run {date_str}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
         "--allow-empty"],
        ["git", "push", "origin", "main"],
    ]
    for cmd in mf_cmds:
        result = subprocess.run(cmd, cwd=str(MIROFISH_DIR), capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        log.info(f"  mirofish git {cmd[1]}: {result.returncode}")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_scrape():
    log.info("=== Step 1: Scrape all data ===")
    run("scrape_oil.py")
    run("scrape_interest_rates.py")
    run("scrape_economic.py")
    run("scrape_daily_indicators.py")
    run("scrape_creb.py")
    # Note: scrape_creb_historical.py is slow (Wayback Machine) — skip on routine runs


def step_merge():
    log.info("=== Step 2: Merge datasets ===")
    run("merge_datasets.py")


def step_models():
    log.info("=== Step 3: Retrain models ===")
    run("model_xgboost.py")
    run("model_monthly.py")
    run("model_daily.py")


def step_news():
    log.info("=== Step 4: Scrape news ===")
    run("scrape_news.py")


def step_report():
    log.info("=== Step 5: Generate market report ===")
    run("generate_report.py")


def step_mirofish():
    log.info("=== Step 6-7: MiroFish simulation ===")
    start_mirofish()
    run("run_mirofish_pipeline.py")
    run("translate_report.py")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start = datetime.now()
    log.info("=" * 60)
    log.info(f"Calgary Housing Pipeline starting: {start.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    failed_steps = []

    steps = [
        ("scrape",    step_scrape),
        ("merge",     step_merge),
        ("models",    step_models),
        ("news",      step_news),
        ("report",    step_report),
        ("mirofish",  step_mirofish),
    ]

    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            log.error(f"Step '{name}' failed: {e}")
            failed_steps.append(name)
            # Don't abort — continue with remaining steps

    # Always commit whatever we have
    try:
        git_commit_and_push()
    except Exception as e:
        log.error(f"Git push failed: {e}")

    elapsed = (datetime.now() - start).total_seconds() / 60
    log.info("=" * 60)
    if failed_steps:
        log.warning(f"Pipeline finished with failures in: {failed_steps}")
    else:
        log.info("Pipeline complete — all steps succeeded.")
    log.info(f"Total time: {elapsed:.1f} minutes")
    log.info(f"Log: {log_file}")
    log.info("=" * 60)

    return 1 if failed_steps else 0


if __name__ == "__main__":
    sys.exit(main())
