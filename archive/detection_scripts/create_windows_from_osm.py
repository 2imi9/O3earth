#!/usr/bin/env python3
"""
create_windows_from_osm.py — Generate rslearn windows at OSM energy polygon locations.

We have 1.66M OSM energy polygons but only ~600 training windows.
This script creates windows centered on clusters of OSM features to
massively expand the training dataset.

Strategy:
1. Load OSM polygons from GeoJSON
2. Cluster nearby polygons into grid cells (0.05° ≈ 5.5km)
3. Filter to cells that contain actual energy infrastructure
4. Create rslearn windows at each cluster centroid
5. Focus on solar + wind (classes with most OSM data)

Usage:
    python scripts/create_windows_from_osm.py --root ./dataset --max-windows 5000
    python scripts/create_windows_from_osm.py --root ./dataset --max-windows 2000 --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict


def load_osm_clusters(geojson_path, cell_deg=0.05, class_filter=None):
    """
    Load OSM polygons and cluster by grid cell.

    Args:
        geojson_path: path to osm_energy_polygons.geojson
        cell_deg: grid cell size in degrees (~5.5km at mid-latitudes)
        class_filter: set of class names to include (e.g., {"solar", "wind"})

    Returns:
        list of dicts with {lon, lat, count, classes, total_area_m2}
    """
    print(f"Loading OSM polygons from {geojson_path}...")
    t0 = time.time()

    with open(geojson_path) as f:
        data = json.load(f)

    print(f"  Loaded {len(data['features'])} features in {time.time()-t0:.1f}s")

    # Cluster by grid cell
    cells = defaultdict(lambda: {"lons": [], "lats": [], "classes": defaultdict(int), "count": 0})

    skipped = 0
    for feat in data["features"]:
        props = feat.get("properties", {})
        class_name = props.get("class_name", "")

        if class_filter and class_name not in class_filter:
            skipped += 1
            continue

        # Get centroid from geometry
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        # Extract a representative point
        try:
            if geom["type"] == "Polygon":
                # Use first coordinate of exterior ring
                lon, lat = coords[0][0][0], coords[0][0][1]
            elif geom["type"] == "MultiPolygon":
                lon, lat = coords[0][0][0][0], coords[0][0][0][1]
            elif geom["type"] == "Point":
                lon, lat = coords[0], coords[1]
            else:
                continue
        except (IndexError, KeyError, TypeError):
            continue

        # Filter to continental US
        if not (-125 <= lon <= -66 and 24 <= lat <= 50):
            continue

        key = (round(lon / cell_deg), round(lat / cell_deg))
        cells[key]["lons"].append(lon)
        cells[key]["lats"].append(lat)
        cells[key]["classes"][class_name] += 1
        cells[key]["count"] += 1

    print(f"  Skipped {skipped} features (class filter)")

    # Convert to list with centroids
    clusters = []
    for key, cell in cells.items():
        centroid_lon = sum(cell["lons"]) / len(cell["lons"])
        centroid_lat = sum(cell["lats"]) / len(cell["lats"])
        clusters.append({
            "lon": centroid_lon,
            "lat": centroid_lat,
            "count": cell["count"],
            "classes": dict(cell["classes"]),
            "primary_class": max(cell["classes"], key=cell["classes"].get),
        })

    # Sort by count descending (prioritize areas with most infrastructure)
    clusters.sort(key=lambda c: c["count"], reverse=True)

    print(f"  Found {len(clusters)} unique clusters")
    class_totals = defaultdict(int)
    for c in clusters:
        for cls, cnt in c["classes"].items():
            class_totals[cls] += cnt
    for cls, cnt in sorted(class_totals.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {cnt} polygons")

    return clusters


def create_windows(clusters, dataset_root, max_windows, group="default",
                   half_deg=0.02, grid_size=128, resolution=10,
                   start="2023-05-01T00:00:00+00:00",
                   end="2023-09-01T00:00:00+00:00",
                   dry_run=False):
    """
    Create rslearn windows at cluster locations.

    Args:
        clusters: list from load_osm_clusters()
        dataset_root: path to rslearn dataset
        max_windows: max number of cluster windows to create
        group: rslearn window group
        half_deg: half-width of bounding box in degrees
                  0.02° ≈ 2.2km → gives ~440m x 440m at 128px x 10m
                  0.05° ≈ 5.5km → gives ~1.1km x 1.1km
        grid_size: pixel size of each window
        resolution: meters per pixel
        start/end: temporal range for imagery
        dry_run: if True, just print what would happen
    """
    selected = clusters[:max_windows]

    # Stats
    class_counts = defaultdict(int)
    for c in selected:
        class_counts[c["primary_class"]] += 1

    print(f"\nCreating {len(selected)} windows (from {len(clusters)} clusters):")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt} clusters")

    if dry_run:
        print("\n[DRY RUN] Would create windows at these locations:")
        for i, c in enumerate(selected[:20]):
            print(f"  {i+1}. ({c['lon']:.4f}, {c['lat']:.4f}) — {c['count']} {c['primary_class']} polygons")
        if len(selected) > 20:
            print(f"  ... and {len(selected)-20} more")
        return

    # Create windows in batches to avoid command line limits
    success = 0
    failed = 0
    batch_size = 50

    for i in range(0, len(selected), batch_size):
        batch = selected[i:i+batch_size]
        print(f"\nBatch {i//batch_size + 1}/{(len(selected)-1)//batch_size + 1} "
              f"({len(batch)} windows)...")

        for j, cluster in enumerate(batch):
            lon, lat = cluster["lon"], cluster["lat"]
            bbox = f"{lon-half_deg},{lat-half_deg},{lon+half_deg},{lat+half_deg}"

            cmd = [
                "rslearn", "dataset", "add_windows",
                "--root", str(dataset_root),
                "--group", group,
                "--utm", "--resolution", str(resolution),
                "--src_crs", "EPSG:4326",
                f"--box={bbox}",
                "--start", start,
                "--end", end,
                "--grid_size", str(grid_size),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    success += 1
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"  WARN: Failed at ({lon:.4f}, {lat:.4f}): {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                failed += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  ERROR: {e}")

            # Progress
            total = i + j + 1
            if total % 100 == 0:
                print(f"  Progress: {total}/{len(selected)} ({success} ok, {failed} fail)")

    print(f"\nDone: {success} windows created, {failed} failed")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Create rslearn windows at OSM energy polygon locations"
    )
    parser.add_argument("--root", default="./dataset", help="rslearn dataset root")
    parser.add_argument("--osm-path", default="data/osm_energy_polygons.geojson",
                        help="Path to OSM GeoJSON")
    parser.add_argument("--max-windows", type=int, default=3000,
                        help="Max windows to create (default 3000)")
    parser.add_argument("--cell-deg", type=float, default=0.05,
                        help="Clustering grid cell size in degrees (default 0.05)")
    parser.add_argument("--half-deg", type=float, default=0.02,
                        help="Half-width of window bbox in degrees (default 0.02)")
    parser.add_argument("--group", default="default",
                        help="rslearn window group (default: default)")
    parser.add_argument("--classes", nargs="+", default=["solar", "wind"],
                        help="Energy classes to include (default: solar wind)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would happen")
    parser.add_argument("--grid-size", type=int, default=128,
                        help="Window grid size in pixels (default 128)")

    args = parser.parse_args()

    # Load and cluster OSM polygons
    clusters = load_osm_clusters(
        args.osm_path,
        cell_deg=args.cell_deg,
        class_filter=set(args.classes),
    )

    if not clusters:
        print("No clusters found. Check --osm-path and --classes.")
        sys.exit(1)

    # Create windows
    create_windows(
        clusters,
        dataset_root=args.root,
        max_windows=args.max_windows,
        group=args.group,
        half_deg=args.half_deg,
        grid_size=args.grid_size,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"  1. rslearn dataset prepare --root {args.root} --workers 16")
        print(f"  2. rslearn dataset ingest --root {args.root} --workers 16")
        print(f"  3. rslearn dataset materialize --root {args.root} --workers 16 --ignore-errors")
        print(f"  4. python scripts/relabel_osm.py --root {args.root}")
        print(f"  5. python scripts/balance_dataset.py")
        print(f"  6. rslearn model fit --config configs/model.yaml")


if __name__ == "__main__":
    main()
