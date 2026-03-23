"""Overnight experiment runner — Runs 23, 24, 25 sequentially."""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Use venv python
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = sys.executable  # fallback

EXPERIMENTS = [
    {
        "run_name": "run_23_highlr",
        "config": "configs/model_v12_highlr.yaml",
        "description": "Higher LR (0.003) + stronger weights [0.1, 15, 20] + gaussian noise aug",
    },
    {
        "run_name": "run_24_deeper_decoder",
        "config": "configs/model_v13_deeper_decoder.yaml",
        "description": "Deeper decoder (768/384/192, 3 conv layers) + augmentation",
    },
    {
        "run_name": "run_25_extreme_weights",
        "config": "configs/model_v14_focal_loss.yaml",
        "description": "Extreme weights [0.05, 20, 25] + macro F1 + LR 0.002",
    },
]

LOG_FILE = PROJECT_ROOT / "overnight_results.txt"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_experiment(exp):
    run_name = exp["run_name"]
    config = exp["config"]

    log(f"=" * 60)
    log(f"STARTING: {run_name}")
    log(f"Config: {config}")
    log(f"Description: {exp['description']}")
    log(f"=" * 60)

    env = os.environ.copy()
    env["PROJECT_NAME"] = "openenergyengine"
    env["RUN_NAME"] = run_name
    env["MANAGEMENT_DIR"] = "./project_data"
    env["WANDB_MODE"] = "offline"
    env["RSLEARN_MULTIPROCESSING_CONTEXT"] = "spawn"

    start = time.time()

    # Training
    log(f"[{run_name}] Starting training...")
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "rslearn.main", "model", "fit", "--config", config],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    elapsed = (time.time() - start) / 60
    log(f"[{run_name}] Training finished in {elapsed:.1f} min (exit code {result.returncode})")

    if result.returncode != 0:
        log(f"[{run_name}] STDERR (last 500 chars): {result.stderr[-500:]}")
        return False

    # Testing on best checkpoint
    log(f"[{run_name}] Running test on best checkpoint...")
    ckpt_path = PROJECT_ROOT / "project_data" / "openenergyengine" / run_name / "best.ckpt"
    if not ckpt_path.exists():
        log(f"[{run_name}] WARNING: No best.ckpt found, skipping test")
        return True

    test_result = subprocess.run(
        [
            str(VENV_PYTHON), "-m", "rslearn.main", "model", "test",
            "--config", config,
            "--ckpt_path", str(ckpt_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    log(f"[{run_name}] Test exit code: {test_result.returncode}")

    # Parse test output for metrics
    for line in test_result.stdout.split("\n"):
        if "F1" in line or "precision" in line or "recall" in line or "accuracy" in line or "loss" in line:
            log(f"[{run_name}] {line.strip()}")

    # Also check stderr for metrics (Lightning sometimes logs there)
    for line in test_result.stderr.split("\n"):
        if "F1" in line or "precision" in line or "recall" in line or "test_" in line:
            log(f"[{run_name}] {line.strip()}")

    # Read CSV metrics
    csv_candidates = list((PROJECT_ROOT / "project_data" / "openenergyengine" / run_name / "lightning_logs").rglob("metrics.csv"))
    if csv_candidates:
        latest_csv = sorted(csv_candidates)[-1]
        log(f"[{run_name}] Metrics CSV: {latest_csv}")
        with open(latest_csv) as f:
            lines = f.readlines()
            if len(lines) > 1:
                log(f"[{run_name}] CSV header: {lines[0].strip()}")
                log(f"[{run_name}] CSV last row: {lines[-1].strip()}")
                # Find best val F1
                best_f1 = 0
                best_epoch = -1
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    # Find val F1 column
                    header = lines[0].strip().split(",")
                    for i, h in enumerate(header):
                        if "val" in h and "F1" in h:
                            try:
                                f1 = float(parts[i]) if parts[i] else 0
                                if f1 > best_f1:
                                    best_f1 = f1
                                    epoch_idx = header.index("epoch") if "epoch" in header else 0
                                    best_epoch = parts[epoch_idx] if parts[epoch_idx] else "?"
                            except (ValueError, IndexError):
                                pass
                log(f"[{run_name}] BEST VAL F1: {best_f1:.4f} at epoch {best_epoch}")

    return True


def main():
    log("=" * 60)
    log("OVERNIGHT EXPERIMENT RUNNER — OpenEnergy-Engine")
    log(f"Running {len(EXPERIMENTS)} experiments")
    log("=" * 60)

    results = {}
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        results[exp["run_name"]] = "OK" if success else "FAILED"
        log(f"[{exp['run_name']}] Result: {'OK' if success else 'FAILED'}")
        log("")

    log("=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    for name, status in results.items():
        log(f"  {name}: {status}")
    log("=" * 60)


if __name__ == "__main__":
    main()
