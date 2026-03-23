#!/usr/bin/env python3
"""
overnight_full.py — Full overnight pipeline:
  1. Relabel ALL 9,292 ready windows with OSM polygons
  2. Balance dataset (equal energy + bg)
  3. Train frozen encoder for 100 epochs
  4. Test best checkpoint
"""

import subprocess
import sys
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime

VENV_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
ROOT = str(Path(__file__).resolve().parent.parent)
LOG_FILE = Path(ROOT) / "overnight_full_results.txt"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, timeout=None, desc=""):
    log(f"  Running: {desc or ' '.join(cmd[:3])}")
    log(f"  Command: {' '.join(cmd)}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=ROOT
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            log(f"  FAILED ({elapsed:.0f}s): {result.stderr[-500:]}")
            return False, result
        log(f"  Done in {elapsed/60:.1f}min")
        return True, result
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout}s")
        return False, None


def step1_relabel():
    """Relabel all ready windows with OSM polygons."""
    log("=" * 60)
    log("STEP 1: Relabel ALL ready windows with OSM polygons")
    log("=" * 60)

    ok, result = run_cmd(
        [VENV_PYTHON, "scripts/relabel_osm.py",
         "--root", "./dataset", "--osm-path", "data/osm_energy_polygons.geojson"],
        timeout=7200,
        desc="Relabel all 9292 ready windows"
    )
    if ok:
        log(f"  Output tail: {result.stdout[-1000:]}")
    return ok


def step2_balance():
    """Balance dataset: assign train/val splits with proper energy/bg ratio."""
    log("=" * 60)
    log("STEP 2: Balance dataset and assign train/val splits")
    log("=" * 60)

    import numpy as np

    ready_dir = Path(ROOT) / "dataset" / "windows" / "ready"
    windows = sorted([d for d in ready_dir.iterdir() if d.is_dir()])

    energy_windows = []
    bg_windows = []

    for w in windows:
        label_path = w / "layers" / "label_raster" / "label" / "geotiff.tif"
        if not label_path.exists():
            continue

        try:
            import tifffile
            label = tifffile.imread(str(label_path))
            uniq = np.unique(label)
            if 1 in uniq or 2 in uniq:
                energy_windows.append(w)
            else:
                bg_windows.append(w)
        except Exception:
            bg_windows.append(w)

    log(f"  Energy windows: {len(energy_windows)}")
    log(f"  BG-only windows: {len(bg_windows)}")

    # Use 2:1 ratio of bg:energy (or all bg if fewer)
    np.random.seed(42)
    n_bg = min(len(bg_windows), len(energy_windows) * 2)
    selected_bg = list(np.random.choice(len(bg_windows), size=n_bg, replace=False))
    selected_bg_windows = [bg_windows[i] for i in selected_bg]

    all_selected = energy_windows + selected_bg_windows
    log(f"  Selected for training: {len(energy_windows)} energy + {n_bg} bg = {len(all_selected)}")

    # Hash-based train/val split (80/20)
    train_windows = []
    val_windows = []
    for w in all_selected:
        h = int(hashlib.md5(w.name.encode()).hexdigest(), 16) % 100
        if h < 80:
            train_windows.append(w)
        else:
            val_windows.append(w)

    log(f"  Train: {len(train_windows)}, Val: {len(val_windows)}")

    # Write splits to metadata
    for w in windows:
        meta_path = w / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if "options" not in meta:
            meta["options"] = {}

        if w in train_windows:
            meta["options"]["split"] = "train"
        elif w in val_windows:
            meta["options"]["split"] = "val"
        else:
            meta["options"]["split"] = "train_excluded"

        meta_path.write_text(json.dumps(meta, indent=2))

    log(f"  Splits assigned successfully")
    return True


def step3_train(run_name, max_epochs=100):
    """Train with frozen encoder for many epochs."""
    log("=" * 60)
    log(f"STEP 3: Train run '{run_name}' — {max_epochs} epochs, frozen encoder")
    log("=" * 60)

    import os
    env = os.environ.copy()
    env["PROJECT_NAME"] = "openenergyengine"
    env["RUN_NAME"] = run_name
    env["MANAGEMENT_DIR"] = "./project_data"
    env["WANDB_MODE"] = "offline"
    env["RSLEARN_MULTIPROCESSING_CONTEXT"] = "spawn"  # Windows fix

    # Create config with more epochs
    config_path = Path(ROOT) / "configs" / f"model_{run_name}.yaml"
    base_config = Path(ROOT) / "configs" / "model_v11_fullfreeze.yaml"
    config_text = base_config.read_text()

    # Update max_epochs
    config_text = config_text.replace("max_epochs: 30", f"max_epochs: {max_epochs}")
    # More patience for early stopping since we have more data
    config_text = config_text.replace("patience: 8", "patience: 15")
    # Check val more frequently with bigger dataset
    config_text = config_text.replace("check_val_every_n_epoch: 2", "check_val_every_n_epoch: 1")

    config_path.write_text(config_text)
    log(f"  Config: {config_path}")

    start = time.time()
    try:
        result = subprocess.run(
            [VENV_PYTHON, "-m", "rslearn.main", "model", "fit",
             "--config", str(config_path)],
            capture_output=True, text=True,
            timeout=36000,  # 10 hour timeout
            cwd=ROOT, env=env
        )
        elapsed = time.time() - start
        log(f"  Training completed in {elapsed/3600:.1f}h (exit={result.returncode})")

        # Extract key metrics from output
        for line in result.stdout.split("\n")[-50:]:
            if any(k in line.lower() for k in ["f1", "val_loss", "epoch", "best"]):
                log(f"  {line.strip()}")

        if result.returncode != 0:
            log(f"  STDERR tail: {result.stderr[-500:]}")

        return result.returncode == 0, result
    except subprocess.TimeoutExpired:
        log(f"  Training TIMEOUT after 10h")
        return False, None


def step4_test(run_name):
    """Test best checkpoint."""
    log("=" * 60)
    log(f"STEP 4: Test best checkpoint for '{run_name}'")
    log("=" * 60)

    import os
    env = os.environ.copy()
    env["PROJECT_NAME"] = "openenergyengine"
    env["RUN_NAME"] = run_name
    env["MANAGEMENT_DIR"] = "./project_data"
    env["WANDB_MODE"] = "offline"
    env["RSLEARN_MULTIPROCESSING_CONTEXT"] = "spawn"  # Windows fix

    config_path = Path(ROOT) / "configs" / f"model_{run_name}.yaml"

    ckpt_path = Path(ROOT) / "project_data" / "openenergyengine" / run_name / "best.ckpt"
    if not ckpt_path.exists():
        log(f"  No best.ckpt found at {ckpt_path}")
        return False

    ok, result = run_cmd(
        [VENV_PYTHON, "-m", "rslearn.main", "model", "test",
         "--config", str(config_path),
         "--ckpt_path", str(ckpt_path)],
        timeout=3600,
        desc=f"Test {run_name}"
    )
    if ok and result:
        log(f"  Test output:\n{result.stdout[-2000:]}")
    return ok


def main():
    log("=" * 60)
    log("OVERNIGHT FULL PIPELINE — OpenEnergy-Engine")
    log(f"Started at {datetime.now()}")
    log("=" * 60)

    # Step 1: Relabel (already done — 340 solar, 9 wind)
    # step1_relabel()

    # Step 2: Balance (already done — 855 train, 192 val)
    # step2_balance()

    # Step 3: Train
    run_name = "run_overnight_v1"
    ok, _ = step3_train(run_name, max_epochs=100)

    # Step 4: Test (even if training had issues, try to test best ckpt)
    step4_test(run_name)

    log("=" * 60)
    log("PIPELINE COMPLETE")
    log(f"Finished at {datetime.now()}")
    log("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
