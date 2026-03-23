#!/usr/bin/env python3
"""
build_large_dataset.py — Full pipeline to build a large training dataset.

Steps:
1. Create windows from OSM polygon clusters (3000+ locations)
2. Run rslearn prepare (query Sentinel-2 metadata)
3. Run rslearn ingest (download satellite imagery)
4. Run rslearn materialize (crop/reproject into tiles)
5. Move prepared windows to 'ready' group
6. Relabel with OSM polygons
7. Split train/val
8. Balance dataset

Usage:
    python scripts/build_large_dataset.py --max-windows 3000 --workers 8
    python scripts/build_large_dataset.py --max-windows 5000 --workers 16  # cloud GPU
    python scripts/build_large_dataset.py --skip-to relabel  # resume from step 6
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd, desc, timeout=None, check=True):
    """Run a command with logging."""
    log(f"  Running: {desc}")
    log(f"  Command: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env={
            **__import__("os").environ,
            "RSLEARN_MULTIPROCESSING_CONTEXT": "spawn",
        },
    )
    elapsed = time.time() - t0
    log(f"  Done in {elapsed/60:.1f}min (exit={result.returncode})")
    if result.returncode != 0 and check:
        log(f"  STDERR: {result.stderr[-500:]}")
    return result


def step_create_windows(args):
    """Step 1: Create windows from OSM clusters."""
    log("=" * 60)
    log("STEP 1: Create windows from OSM polygon locations")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "scripts/create_windows_from_osm.py",
        "--root", args.root,
        "--osm-path", "data/osm_energy_polygons.geojson",
        "--max-windows", str(args.max_windows),
        "--classes", "solar", "wind",
        "--group", "default",
    ]
    return run_cmd(cmd, f"Creating up to {args.max_windows} windows", timeout=3600)


def step_prepare(args):
    """Step 2: Query Sentinel-2 metadata."""
    log("=" * 60)
    log("STEP 2: Prepare dataset (query satellite metadata)")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "-m", "rslearn.main",
        "dataset", "prepare",
        "--root", args.root,
        "--workers", str(args.workers),
    ]
    return run_cmd(cmd, "rslearn dataset prepare", timeout=7200)


def step_ingest(args):
    """Step 3: Download satellite imagery."""
    log("=" * 60)
    log("STEP 3: Ingest dataset (download satellite imagery)")
    log("  This is the slowest step — downloads from Planetary Computer")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "-m", "rslearn.main",
        "dataset", "ingest",
        "--root", args.root,
        "--workers", str(args.workers),
    ]
    return run_cmd(cmd, "rslearn dataset ingest", timeout=36000)  # 10 hours max


def step_materialize(args):
    """Step 4: Crop and reproject into tiles."""
    log("=" * 60)
    log("STEP 4: Materialize dataset (crop/reproject tiles)")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "-m", "rslearn.main",
        "dataset", "materialize",
        "--root", args.root,
        "--workers", str(args.workers),
        "--ignore-errors",
    ]
    return run_cmd(cmd, "rslearn dataset materialize", timeout=14400)


def step_move_to_ready(args):
    """Step 5: Move prepared windows to 'ready' group."""
    log("=" * 60)
    log("STEP 5: Move materialized windows to 'ready' group")
    log("=" * 60)

    # Check which default windows have sentinel2 data
    default_dir = Path(args.root) / "windows" / "default"
    ready_dir = Path(args.root) / "windows" / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)

    if not default_dir.exists():
        log("  No default windows found")
        return

    moved = 0
    skipped = 0
    for window_dir in default_dir.iterdir():
        if not window_dir.is_dir():
            continue
        # Check if it has materialized sentinel2 data
        s2_dir = window_dir / "layers" / "sentinel2_l2a"
        if s2_dir.exists() and any(s2_dir.rglob("*.tif")):
            dest = ready_dir / window_dir.name
            if not dest.exists():
                import shutil
                shutil.move(str(window_dir), str(dest))
                moved += 1
            else:
                skipped += 1
        else:
            skipped += 1

    log(f"  Moved {moved} windows to ready/, skipped {skipped}")


def step_relabel(args):
    """Step 6: Relabel with OSM polygons."""
    log("=" * 60)
    log("STEP 6: Relabel windows with OSM polygon boundaries")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "scripts/relabel_osm.py",
        "--root", args.root,
        "--osm-path", "data/osm_energy_polygons.geojson",
    ]
    return run_cmd(cmd, "Relabeling with OSM polygons", timeout=7200)


def step_split(args):
    """Step 7: Split train/val."""
    log("=" * 60)
    log("STEP 7: Split into train/val")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "scripts/split_train_val.py",
        "--root", args.root,
    ]
    return run_cmd(cmd, "Train/val split", timeout=600, check=False)


def step_balance(args):
    """Step 8: Balance dataset."""
    log("=" * 60)
    log("STEP 8: Balance dataset (reduce background-only windows)")
    log("=" * 60)

    cmd = [
        VENV_PYTHON, "scripts/balance_dataset.py",
    ]
    return run_cmd(cmd, "Balancing dataset", timeout=600, check=False)


def step_stats(args):
    """Print dataset statistics."""
    log("=" * 60)
    log("DATASET STATISTICS")
    log("=" * 60)

    ready_dir = Path(args.root) / "windows" / "ready"
    if not ready_dir.exists():
        log("  No ready windows")
        return

    total = 0
    has_label = 0
    has_s2 = 0
    splits = Counter()

    for window_dir in ready_dir.iterdir():
        if not window_dir.is_dir():
            continue
        total += 1

        if (window_dir / "layers" / "sentinel2_l2a").exists():
            has_s2 += 1
        if (window_dir / "layers" / "label_raster").exists():
            has_label += 1

        meta_path = window_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            split = meta.get("options", {}).get("split", "none")
            splits[split] += 1

    log(f"  Total windows: {total}")
    log(f"  With Sentinel-2: {has_s2}")
    log(f"  With labels: {has_label}")
    log(f"  Splits: {dict(splits)}")


STEPS = {
    "windows": step_create_windows,
    "prepare": step_prepare,
    "ingest": step_ingest,
    "materialize": step_materialize,
    "move": step_move_to_ready,
    "relabel": step_relabel,
    "split": step_split,
    "balance": step_balance,
    "stats": step_stats,
}

STEP_ORDER = ["windows", "prepare", "ingest", "materialize", "move", "relabel", "split", "balance", "stats"]


def main():
    parser = argparse.ArgumentParser(description="Build large training dataset")
    parser.add_argument("--root", default="./dataset", help="rslearn dataset root")
    parser.add_argument("--max-windows", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-to", choices=STEP_ORDER,
                        help="Skip to a specific step (e.g., --skip-to relabel)")
    parser.add_argument("--only", choices=STEP_ORDER,
                        help="Run only one step")

    args = parser.parse_args()

    log("=" * 60)
    log("BUILD LARGE DATASET — OpenEnergy-Engine")
    log(f"Max windows: {args.max_windows}, Workers: {args.workers}")
    log("=" * 60)

    if args.only:
        STEPS[args.only](args)
        return

    start_idx = 0
    if args.skip_to:
        start_idx = STEP_ORDER.index(args.skip_to)
        log(f"Skipping to step: {args.skip_to}")

    for step_name in STEP_ORDER[start_idx:]:
        try:
            STEPS[step_name](args)
        except Exception as e:
            log(f"FAILED at step {step_name}: {e}")
            log("You can resume with: --skip-to <next_step>")
            break

    log("\nPipeline complete!")


if __name__ == "__main__":
    main()
