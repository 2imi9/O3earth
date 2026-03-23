#!/usr/bin/env python3
"""
Create label rasters for existing rslearn windows using EIA 860 plant data.

This follows the official olmoearth pattern (like mozambique_lulc / satlas_solar_farm):
  1. Read each window's spatial extent from metadata.json
  2. Find EIA plants that fall within the window
  3. Rasterize plant class IDs into a GeoTIFF label
  4. Save as layers/label_raster/label/geotiff.tif

Usage:
  python scripts/create_label_rasters.py \
    --csv source_data/eia860/3_1_Generator_Y2024_merged.csv \
    --root dataset \
    --group ready \
    --buffer 500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_bounds

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_sources.eia_860 import load_eia860_plants, CLASS_NAMES


def parse_window_metadata(window_dir: Path):
    """Parse window metadata to get CRS, bounds, and size."""
    meta_path = window_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # rslearn metadata format
    projection = meta.get("projection", {})
    crs = projection.get("crs", None)
    x_res = projection.get("x_resolution", 10.0)
    y_res = projection.get("y_resolution", -10.0)

    bounds = meta.get("bounds", None)  # [col_min, row_min, col_max, row_max] in pixel space
    if not crs or not bounds:
        return None

    # Convert pixel bounds to CRS coordinates
    col_min, row_min, col_max, row_max = bounds
    # In rslearn: crs_x = pixel_col * x_resolution, crs_y = pixel_row * y_resolution
    x_min = col_min * x_res
    x_max = col_max * x_res
    # y_res is negative, so row_min * y_res > row_max * y_res
    y_min = row_max * y_res  # bottom (smaller y)
    y_max = row_min * y_res  # top (larger y)
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    width = col_max - col_min   # pixels
    height = row_max - row_min  # pixels

    return {
        "crs": crs,
        "x_res": x_res,
        "y_res": y_res,
        "bounds_px": bounds,  # [col_min, row_min, col_max, row_max]
        "bounds_crs": [x_min, y_min, x_max, y_max],  # [xmin, ymin, xmax, ymax]
        "width": width,
        "height": height,
    }


def create_label_raster(window_dir: Path, plants, meta, buffer_meters: float):
    """Create a label raster GeoTIFF for a window."""
    crs = meta["crs"]
    width = meta["width"]
    height = meta["height"]
    x_min, y_min, x_max, y_max = meta["bounds_crs"]
    col_min, row_min, col_max, row_max = meta["bounds_px"]
    x_res = abs(meta["x_res"])
    y_res_abs = abs(meta["y_res"])

    # Create empty label raster
    label = np.zeros((height, width), dtype=np.uint8)
    capacity_map = np.zeros((height, width), dtype=np.float32)

    # Transform plant WGS84 coords to window CRS
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    plants_found = 0
    for plant in plants:
        px, py = transformer.transform(plant.lon, plant.lat)

        # Check if plant falls within window bounds (with buffer)
        if (px < x_min - buffer_meters or px > x_max + buffer_meters or
            py < y_min - buffer_meters or py > y_max + buffer_meters):
            continue

        # Convert CRS coords to pixel coords within this window
        col = (px - x_min) / x_res
        row = (y_max - py) / y_res_abs  # y-axis is flipped (top=0)

        buf_pix = buffer_meters / x_res

        c_min = max(0, int(col - buf_pix))
        c_max = min(width, int(col + buf_pix) + 1)
        r_min = max(0, int(row - buf_pix))
        r_max = min(height, int(row + buf_pix) + 1)

        if c_min >= c_max or r_min >= r_max:
            continue

        plants_found += 1
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                dist = ((c - col) ** 2 + (r - row) ** 2) ** 0.5
                if dist <= buf_pix and plant.capacity_mw > capacity_map[r, c]:
                    label[r, c] = plant.energy_class
                    capacity_map[r, c] = plant.capacity_mw

    # Write GeoTIFF
    label_dir = window_dir / "layers" / "label_raster" / "label"
    label_dir.mkdir(parents=True, exist_ok=True)

    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    with rasterio.open(
        str(label_dir / "geotiff.tif"),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(label.reshape(1, height, width))

    # Mark layer as completed (rslearn convention)
    completed_path = window_dir / "layers" / "label_raster" / "completed"
    completed_path.touch()

    return plants_found, int(np.sum(label > 0))


def build_spatial_index(plants, crs_list):
    """Pre-transform all plants to common CRS projections for fast lookup."""
    # Group by UTM zone for efficiency
    plant_coords = {}
    for crs in crs_list:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        coords = []
        for plant in plants:
            px, py = transformer.transform(plant.lon, plant.lat)
            coords.append((px, py))
        plant_coords[crs] = coords
    return plant_coords


def main():
    parser = argparse.ArgumentParser(description="Create label rasters for rslearn windows")
    parser.add_argument("--csv", required=True, help="Path to EIA 860 generator CSV")
    parser.add_argument("--root", default="dataset", help="rslearn dataset root")
    parser.add_argument("--group", default="ready", help="Window group to process")
    parser.add_argument("--buffer", type=float, default=500, help="Buffer radius in meters")
    parser.add_argument("--min-capacity", type=float, default=1.0, help="Min capacity MW")
    parser.add_argument("--only-with-s2", action="store_true", default=True,
                        help="Only process windows that have Sentinel-2 data")
    args = parser.parse_args()

    # Load plants
    print(f"Loading EIA 860 plants from {args.csv}...")
    plants = load_eia860_plants(args.csv, args.min_capacity)
    print(f"  Loaded {len(plants)} plants")

    # Find windows to process
    windows_dir = Path(args.root) / "windows" / args.group
    if not windows_dir.exists():
        print(f"Error: {windows_dir} does not exist")
        sys.exit(1)

    all_windows = sorted([w for w in windows_dir.iterdir() if w.is_dir()])
    print(f"  Found {len(all_windows)} windows in group '{args.group}'")

    if args.only_with_s2:
        all_windows = [w for w in all_windows if (w / "layers" / "sentinel2_l2a").exists()]
        print(f"  {len(all_windows)} windows have Sentinel-2 data")

    if not all_windows:
        print("No windows to process!")
        sys.exit(1)

    # Pre-compute plant coordinates per CRS
    print("Pre-computing plant coordinates per CRS zone...")
    crs_set = set()
    window_metas = {}
    for w in all_windows:
        meta = parse_window_metadata(w)
        if meta:
            crs_set.add(meta["crs"])
            window_metas[w] = meta

    print(f"  {len(crs_set)} unique CRS zones, {len(window_metas)} valid windows")

    # Pre-transform plants per CRS
    plant_coords_by_crs = {}
    for crs in crs_set:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        coords = []
        for plant in plants:
            px, py = transformer.transform(plant.lon, plant.lat)
            coords.append((px, py))
        plant_coords_by_crs[crs] = coords

    # Process windows
    total_with_labels = 0
    total_labeled_pixels = 0
    class_window_counts = {i: 0 for i in range(11)}

    print(f"\nCreating label rasters for {len(window_metas)} windows (buffer={args.buffer}m)...")
    for i, (w, meta) in enumerate(window_metas.items()):
        plants_found, labeled_pixels = create_label_raster(w, plants, meta, args.buffer)

        if plants_found > 0:
            total_with_labels += 1
            total_labeled_pixels += labeled_pixels

            # Count classes
            label_path = w / "layers" / "label_raster" / "label" / "geotiff.tif"
            with rasterio.open(str(label_path)) as ds:
                data = ds.read(1)
                for cls_id in np.unique(data):
                    if cls_id > 0:
                        class_window_counts[int(cls_id)] += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(window_metas):
            print(f"  [{i+1}/{len(window_metas)}] {total_with_labels} windows have labels, "
                  f"{total_labeled_pixels:,} total labeled pixels")

    print(f"\n{'='*60}")
    print(f"Done! Created label rasters for {len(window_metas)} windows.")
    print(f"  Windows with ≥1 labeled pixel: {total_with_labels} ({total_with_labels/len(window_metas)*100:.1f}%)")
    print(f"  Total labeled pixels: {total_labeled_pixels:,}")
    print(f"\n  Class distribution (windows containing each class):")
    for cls_id, count in sorted(class_window_counts.items()):
        if count > 0 or cls_id == 0:
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            print(f"    {cls_id:2d} ({name:12s}): {count:5d} windows")


if __name__ == "__main__":
    main()
