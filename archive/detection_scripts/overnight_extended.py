"""Extended overnight experiment runner — Runs 23-29 sequentially.
Waits for initial batch (runs 23-25) if still running, then continues with 26-29.
"""

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
    VENV_PYTHON = sys.executable

EXPERIMENTS = [
    # Original batch (may already be done)
    {
        "run_name": "run_23_highlr",
        "config": "configs/model_v12_highlr.yaml",
        "description": "Higher LR (0.003) + weights [0.1, 15, 20] + gaussian noise",
        "skip_if_exists": True,
    },
    {
        "run_name": "run_24_deeper_decoder",
        "config": "configs/model_v13_deeper_decoder.yaml",
        "description": "Deeper decoder (768/384/192, 3 conv/res) + augmentation",
        "skip_if_exists": True,
    },
    {
        "run_name": "run_25_extreme_weights",
        "config": "configs/model_v14_focal_loss.yaml",
        "description": "Extreme weights [0.05, 20, 25] + macro F1 + LR 0.002",
        "skip_if_exists": True,
    },
    # Extended batch
    {
        "run_name": "run_26_slow_long",
        "config": "configs/model_v15_warmup_cosine.yaml",
        "description": "Low LR (0.0005) + 60 epochs + patience 15 — let it converge slowly",
        "skip_if_exists": False,
    },
    {
        "run_name": "run_27_large_model",
        "config": "configs/model_v16_large_model.yaml",
        "description": "OlmoEarth LARGE (1024-dim) frozen encoder, batch=4",
        "skip_if_exists": False,
    },
    {
        "run_name": "run_28_batch16",
        "config": "configs/model_v17_batch16.yaml",
        "description": "Batch 16 + grad accum 2 (eff batch 32) + LR 0.002",
        "skip_if_exists": False,
    },
    {
        "run_name": "run_29_lowbg_weight",
        "config": "configs/model_v18_dice_only.yaml",
        "description": "Very low bg weight (0.05) + high energy weights + dice",
        "skip_if_exists": False,
    },
]

LOG_FILE = PROJECT_ROOT / "overnight_results_extended.txt"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_best_f1_from_csv(run_dir):
    """Parse metrics CSV and return best val F1."""
    csv_files = list(run_dir.rglob("metrics.csv"))
    if not csv_files:
        return None, None
    latest_csv = sorted(csv_files)[-1]
    best_f1 = 0
    best_epoch = "?"
    with open(latest_csv) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None, None
        header = lines[0].strip().split(",")
        f1_col = None
        epoch_col = None
        for i, h in enumerate(header):
            if "val" in h and "F1" in h:
                f1_col = i
            if h == "epoch":
                epoch_col = i
        if f1_col is None:
            return None, None
        for line in lines[1:]:
            parts = line.strip().split(",")
            try:
                if parts[f1_col]:
                    f1 = float(parts[f1_col])
                    if f1 > best_f1:
                        best_f1 = f1
                        best_epoch = parts[epoch_col] if epoch_col is not None and parts[epoch_col] else "?"
            except (ValueError, IndexError):
                pass
    return best_f1, best_epoch


def run_experiment(exp):
    run_name = exp["run_name"]
    config = exp["config"]
    run_dir = PROJECT_ROOT / "project_data" / "openenergyengine" / run_name

    # Skip if already completed
    if exp.get("skip_if_exists") and (run_dir / "best.ckpt").exists():
        f1, epoch = get_best_f1_from_csv(run_dir)
        log(f"[{run_name}] SKIPPED — already exists. Best F1={f1:.4f} at epoch {epoch}" if f1 else f"[{run_name}] SKIPPED — already exists")
        return True

    log("=" * 60)
    log(f"STARTING: {run_name}")
    log(f"Config: {config}")
    log(f"Description: {exp['description']}")
    log("=" * 60)

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
        log(f"[{run_name}] STDERR (last 800 chars): {result.stderr[-800:]}")
        return False

    # Test on best checkpoint
    ckpt_path = run_dir / "best.ckpt"
    if not ckpt_path.exists():
        log(f"[{run_name}] WARNING: No best.ckpt, skipping test")
        # Still parse training CSV
        f1, epoch = get_best_f1_from_csv(run_dir)
        if f1:
            log(f"[{run_name}] BEST VAL F1 (from CSV): {f1:.4f} at epoch {epoch}")
        return True

    log(f"[{run_name}] Running test on best checkpoint...")
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

    # Parse test metrics from stderr (Lightning logs there)
    for line in (test_result.stdout + "\n" + test_result.stderr).split("\n"):
        if any(k in line for k in ["F1", "precision", "recall", "accuracy", "test_"]):
            log(f"[{run_name}] {line.strip()}")

    # Parse training CSV for best val metrics
    f1, epoch = get_best_f1_from_csv(run_dir)
    if f1:
        log(f"[{run_name}] BEST VAL F1: {f1:.4f} at epoch {epoch}")

    return True


def main():
    log("=" * 60)
    log("EXTENDED OVERNIGHT RUNNER — OpenEnergy-Engine")
    log(f"Running up to {len(EXPERIMENTS)} experiments")
    log(f"Python: {VENV_PYTHON}")
    log("=" * 60)

    results = {}
    for exp in EXPERIMENTS:
        try:
            success = run_experiment(exp)
            results[exp["run_name"]] = "OK" if success else "FAILED"
        except Exception as e:
            log(f"[{exp['run_name']}] EXCEPTION: {e}")
            results[exp["run_name"]] = f"ERROR: {e}"
        log("")

    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    for name, status in results.items():
        log(f"  {name}: {status}")

    # Print best F1 for each completed run
    log("")
    log("BEST F1 SCORES:")
    for exp in EXPERIMENTS:
        run_dir = PROJECT_ROOT / "project_data" / "openenergyengine" / exp["run_name"]
        f1, epoch = get_best_f1_from_csv(run_dir)
        if f1 and f1 > 0:
            log(f"  {exp['run_name']}: F1={f1:.4f} (epoch {epoch}) — {exp['description']}")
        else:
            log(f"  {exp['run_name']}: no metrics")

    log("=" * 60)
    log("DONE. Good morning Frank!")
    log("=" * 60)


if __name__ == "__main__":
    main()
