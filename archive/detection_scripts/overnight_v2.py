"""Overnight v2 — Conservative experiments based on proven baseline.
Each experiment changes ONLY ONE variable from the proven run 22 config.
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
    {
        "run_name": "run_26_longer",
        "config": "configs/run26_longer_baseline.yaml",
        "description": "Proven baseline + 80 epochs + patience 20 (was 30/8)",
    },
    {
        "run_name": "run_27_lowlr",
        "config": "configs/run27_lower_lr.yaml",
        "description": "Proven baseline + LR 0.0005 (was 0.001)",
    },
    {
        "run_name": "run_28_weights",
        "config": "configs/run28_higher_weights.yaml",
        "description": "Proven baseline + weights [0.1, 15, 20] (was [0.3, 10, 15])",
    },
    {
        "run_name": "run_29_large",
        "config": "configs/run29_large_encoder.yaml",
        "description": "Proven baseline + LARGE encoder 1024-dim (was BASE 768)",
    },
]

LOG_FILE = PROJECT_ROOT / "overnight_results_v2.txt"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def gpu_is_busy():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) >= 2 and int(parts[1].strip()) > 500:
                    return True
        return False
    except Exception:
        return False


def wait_for_gpu():
    if not gpu_is_busy():
        return
    log("GPU busy — waiting...")
    while gpu_is_busy():
        time.sleep(30)
    log("GPU free!")
    time.sleep(5)


def get_best_f1(run_dir):
    csv_files = list(run_dir.rglob("metrics.csv"))
    if not csv_files:
        return 0, "?"
    # Get the version_0 CSV (training metrics), not version_1 (test)
    train_csvs = [f for f in csv_files if "version_0" in str(f)]
    if not train_csvs:
        train_csvs = csv_files
    latest = sorted(train_csvs)[-1]
    best_f1, best_epoch = 0, "?"
    with open(latest) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return 0, "?"
        header = lines[0].strip().split(",")
        f1_col = epoch_col = None
        for i, h in enumerate(header):
            if "val" in h and "F1" in h:
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
                        if epoch_col and len(parts) > epoch_col and parts[epoch_col]:
                            best_epoch = parts[epoch_col]
            except (ValueError, IndexError):
                pass
    return best_f1, best_epoch


def run_experiment(exp):
    run_name = exp["run_name"]
    config = exp["config"]
    run_dir = PROJECT_ROOT / "project_data" / "openenergyengine" / run_name

    if (run_dir / "best.ckpt").exists():
        f1, ep = get_best_f1(run_dir)
        log(f"[{run_name}] SKIP — exists. F1={f1:.4f} (ep {ep})")
        return True, f1

    wait_for_gpu()

    log("=" * 60)
    log(f"START: {run_name} — {exp['description']}")
    log("=" * 60)

    env = os.environ.copy()
    env.update({
        "PROJECT_NAME": "openenergyengine",
        "RUN_NAME": run_name,
        "MANAGEMENT_DIR": "./project_data",
        "WANDB_MODE": "offline",
        "RSLEARN_MULTIPROCESSING_CONTEXT": "spawn",
    })

    t0 = time.time()
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "rslearn.main", "model", "fit", "--config", config],
        env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    mins = (time.time() - t0) / 60
    log(f"[{run_name}] Train done in {mins:.1f}min (exit={result.returncode})")

    if result.returncode != 0:
        log(f"[{run_name}] FAIL: {result.stderr[-600:]}")
        return False, 0

    # Test
    ckpt = run_dir / "best.ckpt"
    if ckpt.exists():
        log(f"[{run_name}] Testing...")
        test = subprocess.run(
            [str(VENV_PYTHON), "-m", "rslearn.main", "model", "test",
             "--config", config, "--ckpt_path", str(ckpt)],
            env=env, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        for line in (test.stdout + "\n" + test.stderr).split("\n"):
            if any(k in line for k in ["F1", "precision", "recall", "test_loss"]):
                c = line.strip()
                if c and len(c) < 200:
                    log(f"[{run_name}] {c}")

    f1, ep = get_best_f1(run_dir)
    log(f"[{run_name}] BEST F1={f1:.4f} (epoch {ep})")
    return True, f1


def main():
    log("=" * 60)
    log("OVERNIGHT v2 — Conservative single-variable experiments")
    log(f"Baseline: run_22 (F1=0.46 peak, weights [0.3,10,15], LR 0.001)")
    log(f"Experiments: {len(EXPERIMENTS)}")
    log("=" * 60)

    results = {}
    for exp in EXPERIMENTS:
        try:
            ok, f1 = run_experiment(exp)
            results[exp["run_name"]] = (ok, f1, exp["description"])
        except Exception as e:
            log(f"[{exp['run_name']}] ERROR: {e}")
            results[exp["run_name"]] = (False, 0, exp["description"])
        log("")

    log("=" * 60)
    log("FINAL RESULTS (baseline: run_22 F1=0.46)")
    log("=" * 60)
    best_name, best_f1 = "run_22_baseline", 0.46
    for name, (ok, f1, desc) in results.items():
        marker = " ★ NEW BEST" if f1 > best_f1 else ""
        log(f"  {name}: F1={f1:.4f} {'OK' if ok else 'FAIL'} — {desc}{marker}")
        if f1 > best_f1:
            best_f1, best_name = f1, name
    log(f"\nBEST OVERALL: {best_name} F1={best_f1:.4f}")
    log("=" * 60)
    log("Good morning Frank!")


if __name__ == "__main__":
    main()
