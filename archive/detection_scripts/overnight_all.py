"""Full overnight experiment runner — waits for GPU, runs all experiments sequentially.
Skips runs that already have a best.ckpt. Safe to run while another training is in progress
(it will wait for GPU to be free).
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

EXPERIMENTS = [
    # Already done from first batch (will be skipped)
    {
        "run_name": "run_23_highlr",
        "config": "configs/model_v12_highlr.yaml",
        "description": "Higher LR (0.003) + weights [0.1, 15, 20] + gaussian noise",
    },
    {
        "run_name": "run_24_deeper_decoder",
        "config": "configs/model_v13_deeper_decoder.yaml",
        "description": "Deeper decoder (768/384/192, 3 conv/res)",
    },
    {
        "run_name": "run_25_extreme_weights",
        "config": "configs/model_v14_focal_loss.yaml",
        "description": "Extreme weights [0.05, 20, 25] + macro F1",
    },
    # New experiments
    {
        "run_name": "run_26_slow_long",
        "config": "configs/model_v15_warmup_cosine.yaml",
        "description": "Low LR (0.0005) + 60 epochs + patience 15",
    },
    {
        "run_name": "run_27_large_model",
        "config": "configs/model_v16_large_model.yaml",
        "description": "OlmoEarth LARGE (1024-dim) frozen encoder, batch=4",
    },
    {
        "run_name": "run_28_batch16",
        "config": "configs/model_v17_batch16.yaml",
        "description": "Batch 16 + grad accum 2 + LR 0.002",
    },
    {
        "run_name": "run_29_lowbg_weight",
        "config": "configs/model_v18_dice_only.yaml",
        "description": "Very low bg weight (0.05) + dice + LR 0.001",
    },
]

LOG_FILE = PROJECT_ROOT / "overnight_results_full.txt"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def gpu_is_busy():
    """Check if another training process is using significant GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    mem_mb = int(parts[1].strip())
                    if mem_mb > 500:  # >500MB = likely training
                        return True
        return False
    except Exception:
        return False


def wait_for_gpu():
    """Wait until GPU is free (no large processes)."""
    if not gpu_is_busy():
        return
    log("GPU is busy — waiting for current training to finish...")
    while gpu_is_busy():
        time.sleep(30)
    log("GPU is now free!")
    time.sleep(5)  # brief cooldown


def get_best_f1_from_csv(run_dir):
    """Parse metrics CSV and return best val F1."""
    csv_files = list(run_dir.rglob("metrics.csv"))
    if not csv_files:
        return 0, "?"
    latest_csv = sorted(csv_files)[-1]
    best_f1 = 0
    best_epoch = "?"
    with open(latest_csv) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return 0, "?"
        header = lines[0].strip().split(",")
        f1_col = epoch_col = None
        for i, h in enumerate(header):
            if "val" in h and "F1" in h:
                f1_col = i
            if "test" in h and "F1" in h and f1_col is None:
                f1_col = i
            if h == "epoch":
                epoch_col = i
        if f1_col is None:
            return 0, "?"
        for line in lines[1:]:
            parts = line.strip().split(",")
            try:
                if len(parts) > f1_col and parts[f1_col]:
                    f1 = float(parts[f1_col])
                    if f1 > best_f1:
                        best_f1 = f1
                        if epoch_col is not None and len(parts) > epoch_col and parts[epoch_col]:
                            best_epoch = parts[epoch_col]
            except (ValueError, IndexError):
                pass
    return best_f1, best_epoch


def run_experiment(exp):
    run_name = exp["run_name"]
    config = exp["config"]
    run_dir = PROJECT_ROOT / "project_data" / "openenergyengine" / run_name

    # Skip if already completed
    if (run_dir / "best.ckpt").exists():
        f1, epoch = get_best_f1_from_csv(run_dir)
        log(f"[{run_name}] SKIPPED — already done. Best F1={f1:.4f} (epoch {epoch})")
        return True, f1

    # Wait for GPU
    wait_for_gpu()

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

    # Train
    log(f"[{run_name}] Training started...")
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "rslearn.main", "model", "fit", "--config", config],
        env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    elapsed = (time.time() - start) / 60
    log(f"[{run_name}] Training done in {elapsed:.1f} min (exit={result.returncode})")

    if result.returncode != 0:
        log(f"[{run_name}] FAILED — {result.stderr[-600:]}")
        return False, 0

    # Test
    ckpt = run_dir / "best.ckpt"
    if ckpt.exists():
        log(f"[{run_name}] Testing best checkpoint...")
        test = subprocess.run(
            [str(VENV_PYTHON), "-m", "rslearn.main", "model", "test",
             "--config", config, "--ckpt_path", str(ckpt)],
            env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        # Log test metrics
        for line in (test.stdout + "\n" + test.stderr).split("\n"):
            if any(k in line for k in ["F1", "precision", "recall", "test_loss"]):
                clean = line.strip()
                if clean and len(clean) < 200:
                    log(f"[{run_name}] {clean}")

    f1, epoch = get_best_f1_from_csv(run_dir)
    log(f"[{run_name}] BEST VAL F1: {f1:.4f} (epoch {epoch})")
    return True, f1


def main():
    log("=" * 60)
    log("FULL OVERNIGHT RUNNER — OpenEnergy-Engine")
    log(f"Experiments: {len(EXPERIMENTS)} | Python: {VENV_PYTHON}")
    log("=" * 60)

    results = {}
    best_overall = ("none", 0)

    for exp in EXPERIMENTS:
        try:
            ok, f1 = run_experiment(exp)
            results[exp["run_name"]] = ("OK" if ok else "FAILED", f1)
            if f1 > best_overall[1]:
                best_overall = (exp["run_name"], f1)
        except Exception as e:
            log(f"[{exp['run_name']}] EXCEPTION: {e}")
            results[exp["run_name"]] = (f"ERROR", 0)
        log("")

    # Final summary
    log("=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    for exp in EXPERIMENTS:
        name = exp["run_name"]
        status, f1 = results.get(name, ("?", 0))
        log(f"  {name}: {status} | F1={f1:.4f} | {exp['description']}")
    log("")
    log(f"BEST RUN: {best_overall[0]} with F1={best_overall[1]:.4f}")
    log("=" * 60)
    log("Good morning Frank! Check overnight_results_full.txt for details.")
    log("=" * 60)


if __name__ == "__main__":
    main()
