#!/usr/bin/env python3
"""
create_windows_from_eia.py — Generate rslearn windows at ALL EIA 860 plant locations.

Maximizes data coverage by creating a window centered on every plant in the
EIA 860 CSV (capacity >= threshold). Nearby plants sharing the same grid cell
are deduplicated so rslearn doesn't create overlapping windows.

Usage:
    python scripts/create_windows_from_eia.py \
        --csv source_data/eia860/3_1_Generator_Y2023.csv \
        --root ./dataset
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root so we can import data_sources
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sources.eia_860 import load_eia860_plants


def cluster_plants(plants, cell_deg=0.05):
    """
    Deduplicate plants by snapping to a grid.

    Each grid cell is ~5.5km at mid-latitudes (0.05 deg).
    Returns one representative (lon, lat) per occupied cell,
    plus counts per cell for logging.
    """
    cells = {}
    for p in plants:
        key = (round(p.lon / cell_deg), round(p.lat / cell_deg))
        if key not in cells:
            cells[key] = {"lon": p.lon, "lat": p.lat, "count": 1, "mw": p.capacity_mw}
        else:
            cells[key]["count"] += 1
            cells[key]["mw"] += p.capacity_mw
    return list(cells.values())


def make_bbox(lon, lat, half_deg=0.25):
    """Create a bounding box string around a point."""
    return f"{lon - half_deg},{lat - half_deg},{lon + half_deg},{lat + half_deg}"


def main():
    parser = argparse.ArgumentParser(
        description="Create rslearn windows at all EIA 860 plant locations"
    )
    parser.add_argument(
        "--csv",
        default="source_data/eia860/3_1_Generator_Y2023.csv",
        help="Path to EIA 860 generator CSV",
    )
    parser.add_argument("--root", default="./dataset", help="rslearn dataset root")
    parser.add_argument("--group", default="default", help="Window group name")
    parser.add_argument(
        "--min-capacity", type=float, default=1.0, help="Min plant capacity (MW)"
    )
    parser.add_argument(
        "--cell-deg",
        type=float,
        default=0.05,
        help="Grid cell size for dedup (degrees, ~5.5km)",
    )
    parser.add_argument(
        "--half-deg",
        type=float,
        default=0.05,
        help="Half-width of bbox around each cluster (degrees, ~5.5km)",
    )
    parser.add_argument(
        "--start", default="2023-05-01", help="Imagery start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2023-09-01", help="Imagery end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--grid-size", type=int, default=128, help="rslearn grid size (pixels)"
    )
    parser.add_argument(
        "--resolution", type=int, default=10, help="Spatial resolution (meters)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: EIA CSV not found at {csv_path}")
        print("Download from: https://www.eia.gov/electricity/data/eia860/")
        print("Save as: source_data/eia860/3_1_Generator_Y2023.csv")
        raise SystemExit(1)

    # Load and cluster plants
    print(f"Loading EIA 860 plants from {csv_path} (min {args.min_capacity} MW)...")
    plants = load_eia860_plants(str(csv_path), min_capacity_mw=args.min_capacity)
    print(f"  Loaded {len(plants)} plants")

    clusters = cluster_plants(plants, cell_deg=args.cell_deg)
    print(f"  Clustered into {len(clusters)} grid cells (cell={args.cell_deg} deg)")

    # Filter to continental US roughly
    conus = [
        c
        for c in clusters
        if -125 < c["lon"] < -66 and 24 < c["lat"] < 50
    ]
    print(f"  {len(conus)} clusters in continental US")

    # Class distribution
    from collections import Counter

    class_counts = Counter()
    for p in plants:
        class_counts[p.energy_class] += 1
    print("\n  Class distribution (plants):")
    from data_sources.eia_860 import CLASS_NAMES

    for cls_id in sorted(class_counts):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"    {cls_id:2d} ({name:12s}): {class_counts[cls_id]:5d} plants")

    # Find rslearn executable — check multiple locations
    rslearn_bin = shutil.which("rslearn")
    if rslearn_bin is None:
        # Try alongside the running Python (venv Scripts/)
        for candidate in [
            Path(sys.executable).parent / "rslearn",
            Path(sys.executable).parent / "rslearn.exe",
            # Also check project .venv
            Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "rslearn.exe",
            Path(__file__).resolve().parent.parent / ".venv" / "bin" / "rslearn",
        ]:
            if candidate.exists():
                rslearn_bin = str(candidate)
                break
    if rslearn_bin is None:
        print("Error: rslearn CLI not found.")
        print("Activate the venv first: .venv\\Scripts\\Activate.ps1")
        raise SystemExit(1)
    print(f"  Using rslearn: {rslearn_bin}")

    # Generate windows
    # Windows doesn't support 'forkserver' — tell rslearn to use 'spawn'
    env = dict(os.environ, RSLEARN_MULTIPROCESSING_CONTEXT="spawn")

    print(f"\nCreating windows for {len(conus)} clusters...")
    success = 0
    fail = 0

    for i, cluster in enumerate(conus):
        bbox = make_bbox(cluster["lon"], cluster["lat"], args.half_deg)
        cmd = [
            rslearn_bin,
            "dataset",
            "add_windows",
            "--root",
            args.root,
            "--group",
            args.group,
            "--utm",
            "--resolution",
            str(args.resolution),
            "--src_crs",
            "EPSG:4326",
            f"--box={bbox}",
            f"--start",
            args.start,
            "--end",
            args.end,
            "--grid_size",
            str(args.grid_size),
        ]

        if args.dry_run:
            print(f"  [{i+1}/{len(conus)}] DRY RUN: {' '.join(cmd)}")
            success += 1
            continue

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            success += 1
            # Print early progress frequently, then every 50
            if (i + 1) <= 5 or (i + 1) % 50 == 0 or i == len(conus) - 1:
                print(f"  [{i+1}/{len(conus)}] {success} ok, {fail} failed", flush=True)
        except subprocess.CalledProcessError as e:
            fail += 1
            if fail <= 5:
                print(f"  [{i+1}/{len(conus)}] FAILED: {e.stderr[:200]}")

    print(f"\nDone. {success} succeeded, {fail} failed out of {len(conus)} clusters.")
    print(f"Windows in {args.root}/windows/{args.group}/")
    print(f"\nNext: bash scripts/build_dataset.sh")


if __name__ == "__main__":
    main()
