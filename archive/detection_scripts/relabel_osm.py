#!/usr/bin/env python3
"""
relabel_osm.py — Replace EIA circle labels with OSM polygon labels.

EIA labels are 500m buffer circles around plant centroids — too noisy for
pixel-level segmentation at 10m Sentinel-2 resolution.

OSM polygons are actual outlines of solar/wind installations, giving
much more accurate pixel-level labels.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import tifffile


CLASS_MAP = {
    "solar": 1,
    "wind": 2,
}


def load_osm_polygons(geojson_path, class_filter=None):
    """Load OSM energy polygons, optionally filtering by class."""
    print(f"Loading OSM polygons from {geojson_path}...")
    with open(geojson_path) as f:
        data = json.load(f)

    polygons = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        class_name = props.get("class_name", "")
        if class_filter and class_name not in class_filter:
            continue
        class_id = CLASS_MAP.get(class_name, 0)
        if class_id == 0:
            continue
        try:
            geom = shape(feat["geometry"])
            if geom.is_valid and not geom.is_empty:
                polygons.append((geom, class_id))
        except Exception:
            continue

    print(f"  Loaded {len(polygons):,} polygons")
    return polygons


def build_spatial_index(polygons):
    """Build a simple grid-based spatial index."""
    from shapely import STRtree
    geoms = [p[0] for p in polygons]
    tree = STRtree(geoms)
    return tree, geoms


def relabel_window(win_dir, polygons, tree, all_geoms, dry_run=False):
    """Relabel a single window using OSM polygons."""
    meta_path = win_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())

    crs = meta["projection"]["crs"]
    bounds = meta["bounds"]  # rslearn scaled coords
    x_res = meta["projection"]["x_resolution"]
    y_res = meta["projection"]["y_resolution"]

    # Get actual UTM bounds from the geotiff
    s2_dir = win_dir / "layers" / "sentinel2_l2a"
    s2_subdirs = sorted([d for d in s2_dir.iterdir() if d.is_dir()]) if s2_dir.exists() else []
    if not s2_subdirs:
        return None

    tif_path = s2_subdirs[0] / "geotiff.tif"
    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as ds:
        utm_bounds = ds.bounds  # left, bottom, right, top
        tif_crs = str(ds.crs)
        tif_transform = ds.transform
        tif_shape = ds.shape

    # Create transformer from WGS84 to window CRS
    to_utm = Transformer.from_crs("EPSG:4326", tif_crs, always_xy=True)
    to_wgs84 = Transformer.from_crs(tif_crs, "EPSG:4326", always_xy=True)

    # Get window bounds in WGS84 for spatial query
    wgs84_bounds = box(
        *to_wgs84.transform(utm_bounds.left, utm_bounds.bottom),
        *to_wgs84.transform(utm_bounds.right, utm_bounds.top),
    )

    # Query spatial index
    candidate_indices = tree.query(wgs84_bounds)
    if len(candidate_indices) == 0:
        # No OSM polygons overlap — label is all background
        label = np.zeros(tif_shape, dtype=np.uint8)
    else:
        # Rasterize matching polygons
        shapes_to_burn = []
        for idx in candidate_indices:
            geom_wgs84 = all_geoms[idx]
            class_id = polygons[idx][1]
            # Transform to UTM
            geom_utm = shapely_transform(
                lambda x, y, z=None: to_utm.transform(x, y),
                geom_wgs84,
            )
            if geom_utm.intersects(box(*utm_bounds)):
                shapes_to_burn.append((geom_utm, class_id))

        if shapes_to_burn:
            label = rasterize(
                shapes_to_burn,
                out_shape=tif_shape,
                transform=tif_transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )
        else:
            label = np.zeros(tif_shape, dtype=np.uint8)

    if dry_run:
        return label

    # Write label as proper georeferenced GeoTIFF
    label_dir = win_dir / "layers" / "label_raster" / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / "geotiff.tif"

    # Back up old label if it exists
    if label_path.exists():
        backup = label_dir / "geotiff.tif.eia_backup"
        if not backup.exists():
            import shutil
            shutil.copy2(str(label_path), str(backup))

    # Write with rasterio to preserve CRS and transform
    with rasterio.open(
        str(label_path),
        "w",
        driver="GTiff",
        height=tif_shape[0],
        width=tif_shape[1],
        count=1,
        dtype=np.uint8,
        crs=tif_crs,
        transform=tif_transform,
    ) as dst:
        dst.write(label, 1)

    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm-path", default="data/osm_energy_polygons.geojson")
    parser.add_argument("--root", default="./dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write labels, just report stats")
    parser.add_argument("--max-windows", type=int, default=0,
                        help="Limit windows to process (0=all)")
    args = parser.parse_args()

    polygons = load_osm_polygons(args.osm_path, class_filter={"solar", "wind"})
    tree, all_geoms = build_spatial_index(polygons)

    windows_dir = Path(args.root) / "windows" / "ready"
    all_windows = sorted([
        d for d in windows_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    if args.max_windows > 0:
        all_windows = all_windows[:args.max_windows]

    print(f"\nProcessing {len(all_windows)} windows...")

    stats = Counter()
    label_pixel_counts = Counter()

    for i, wd in enumerate(all_windows):
        label = relabel_window(wd, polygons, tree, all_geoms, dry_run=args.dry_run)
        if label is None:
            stats["skipped"] += 1
            continue

        unique, counts = np.unique(label, return_counts=True)
        has_solar = 1 in unique
        has_wind = 2 in unique

        if has_solar and has_wind:
            stats["solar+wind"] += 1
        elif has_solar:
            stats["solar"] += 1
        elif has_wind:
            stats["wind"] += 1
        else:
            stats["bg_only"] += 1

        for val, cnt in zip(unique, counts):
            label_pixel_counts[int(val)] += int(cnt)

        if (i + 1) % 500 == 0 or i == len(all_windows) - 1:
            print(f"  {i+1}/{len(all_windows)} — solar:{stats['solar']} wind:{stats['wind']} bg:{stats['bg_only']}")

    print(f"\n{'=' * 50}")
    print(f"Results ({'DRY RUN' if args.dry_run else 'WRITTEN'}):")
    print(f"{'=' * 50}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"\nPixel counts:")
    total_px = sum(label_pixel_counts.values())
    for cls_id in sorted(label_pixel_counts.keys()):
        cnt = label_pixel_counts[cls_id]
        pct = 100.0 * cnt / total_px if total_px > 0 else 0
        cls_name = {0: "background", 1: "solar", 2: "wind"}.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_id} ({cls_name}): {cnt:,} pixels ({pct:.2f}%)")


if __name__ == "__main__":
    main()
