#!/usr/bin/env python3
"""
Create raster labels from OSM energy polygons for rslearn dataset windows.

For each window in dataset/windows/ready/, checks if any OSM polygons overlap.
If they do, rasterizes the polygons into the window's label GeoTIFF.
Windows with no OSM polygon overlap keep their existing EIA point-radius labels.

Optimized with a grid-based spatial index for fast lookups.

Dependencies: json, numpy, tifffile, pyproj, pathlib (all typically installed)
"""

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOJSON_FILE = PROJECT_ROOT / "data" / "osm_energy_polygons.geojson"
DATASET_ROOT = PROJECT_ROOT / "dataset"
WINDOWS_DIR = DATASET_ROOT / "windows" / "ready"
LABEL_SUBPATH = Path("layers") / "label_raster" / "label" / "geotiff.tif"

# Raster dimensions expected for each window
RASTER_W = 128
RASTER_H = 128

# Grid cell size in degrees for spatial index (~10km at mid-latitudes)
GRID_CELL_DEG = 0.1


# ---------------------------------------------------------------------------
# Spatial grid index
# ---------------------------------------------------------------------------


class SpatialGrid:
    """Simple grid-based spatial index in WGS84 coordinates."""

    def __init__(self, cell_size=GRID_CELL_DEG):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _cells_for_bbox(self, bbox):
        """Return set of (col, row) grid cells that a bbox overlaps."""
        lon_min, lat_min, lon_max, lat_max = bbox
        c0 = int(np.floor(lon_min / self.cell_size))
        c1 = int(np.floor(lon_max / self.cell_size))
        r0 = int(np.floor(lat_min / self.cell_size))
        r1 = int(np.floor(lat_max / self.cell_size))
        cells = set()
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                cells.add((c, r))
        return cells

    def insert(self, idx, bbox):
        """Insert a feature index into all overlapping grid cells."""
        for cell in self._cells_for_bbox(bbox):
            self.grid[cell].append(idx)

    def query(self, bbox):
        """Return set of candidate feature indices that might overlap bbox."""
        candidates = set()
        for cell in self._cells_for_bbox(bbox):
            candidates.update(self.grid[cell])
        return candidates


# ---------------------------------------------------------------------------
# Geometry helpers (no shapely dependency)
# ---------------------------------------------------------------------------


def bbox_from_ring(ring: list) -> tuple:
    """Return (min_x, min_y, max_x, max_y) for a coordinate ring."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_overlap(a: tuple, b: tuple) -> bool:
    """Check if two (min_x, min_y, max_x, max_y) bboxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def rasterize_polygon_into(
    raster: np.ndarray,
    ring_projected: list,
    class_id: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    width: int,
    height: int,
):
    """
    Rasterize a single polygon ring into a numpy array.

    Uses scanline approach: for each raster row, find intersections with
    polygon edges and fill between pairs.
    """
    pixel_w = (x_max - x_min) / width
    pixel_h = (y_max - y_min) / height

    # Compute polygon bbox in pixel space to limit iteration
    ring_xs = [p[0] for p in ring_projected]
    ring_ys = [p[1] for p in ring_projected]
    poly_xmin, poly_xmax = min(ring_xs), max(ring_xs)
    poly_ymin, poly_ymax = min(ring_ys), max(ring_ys)

    # Convert to pixel coords (col, row).  Row 0 is at y_max (top of image).
    col_start = max(0, int((poly_xmin - x_min) / pixel_w))
    col_end = min(width - 1, int((poly_xmax - x_min) / pixel_w))
    row_start = max(0, int((y_max - poly_ymax) / pixel_h))
    row_end = min(height - 1, int((y_max - poly_ymin) / pixel_h))

    if col_start > col_end or row_start > row_end:
        return

    n = len(ring_projected)

    for row in range(row_start, row_end + 1):
        # Y coordinate of this row center (top-down: row 0 = y_max)
        y_center = y_max - (row + 0.5) * pixel_h

        # Find x-intersections of scanline with polygon edges
        intersections = []
        j = n - 1
        for i in range(n):
            yi = ring_projected[i][1]
            yj = ring_projected[j][1]
            if (yi > y_center) != (yj > y_center):
                xi = ring_projected[i][0]
                xj = ring_projected[j][0]
                x_int = xi + (y_center - yi) / (yj - yi) * (xj - xi)
                intersections.append(x_int)
            j = i

        intersections.sort()

        # Fill between pairs
        for k in range(0, len(intersections) - 1, 2):
            x_left = intersections[k]
            x_right = intersections[k + 1]
            c_start = max(col_start, int((x_left - x_min) / pixel_w))
            c_end = min(col_end, int((x_right - x_min) / pixel_w))
            for col in range(c_start, c_end + 1):
                raster[row, col] = class_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_osm_features(geojson_path: Path):
    """
    Load GeoJSON features, pre-compute WGS84 bboxes, and build spatial index.
    Returns (features_list, spatial_grid).
    """
    print(f"Loading {geojson_path} ...")
    t0 = time.time()
    with open(geojson_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"  JSON parsed in {time.time() - t0:.1f}s")

    features = data.get("features", [])
    print(f"  {len(features)} total features.")

    # Free the raw dict to save memory
    del data

    # Pre-compute bbox for each feature in WGS84 (lon/lat)
    enriched = []
    grid = SpatialGrid()

    t0 = time.time()
    for f in features:
        geom = f.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates", [])
        if not coords or not coords[0]:
            continue
        ring = coords[0]  # outer ring
        bbox = bbox_from_ring(ring)
        idx = len(enriched)
        enriched.append({"feature": f, "ring_wgs84": ring, "bbox_wgs84": bbox})
        grid.insert(idx, bbox)

    print(f"  {len(enriched)} valid polygons indexed in {time.time() - t0:.1f}s")
    print(f"  Grid cells populated: {len(grid.grid)}")
    return enriched, grid


def load_window_metadata(window_dir: Path) -> dict:
    """Load window metadata.json and return parsed dict, or None."""
    meta_path = window_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pixel_bounds_to_crs(meta: dict) -> tuple:
    """
    Convert rslearn pixel-space bounds to actual CRS coordinates.

    rslearn stores window bounds as [col_min, row_min, col_max, row_max]
    in pixel space.  Multiply by resolution to get CRS coordinates:
      crs_x = col * x_resolution
      crs_y = row * y_resolution   (y_resolution is typically negative)

    Returns (x_min, y_min, x_max, y_max) in the window's CRS.
    """
    bounds = meta.get("bounds", [])
    if len(bounds) != 4:
        return None

    col1, row1, col2, row2 = bounds
    proj = meta.get("projection", {})
    x_res = proj.get("x_resolution", 1.0)
    y_res = proj.get("y_resolution", -1.0)

    # Convert pixel indices to CRS coordinates
    cx1 = col1 * x_res
    cy1 = row1 * y_res
    cx2 = col2 * x_res
    cy2 = row2 * y_res

    return (min(cx1, cx2), min(cy1, cy2), max(cx1, cx2), max(cy1, cy2))


def get_window_bbox_wgs84(meta: dict) -> tuple:
    """
    Convert window bounds to WGS84 bbox for overlap testing.
    Returns (lon_min, lat_min, lon_max, lat_max).
    """
    crs_bounds = pixel_bounds_to_crs(meta)
    if crs_bounds is None:
        return None

    x_min, y_min, x_max, y_max = crs_bounds
    crs = meta.get("projection", {}).get("crs", "EPSG:4326")

    if crs == "EPSG:4326":
        return crs_bounds

    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners = [
            transformer.transform(x_min, y_min),
            transformer.transform(x_min, y_max),
            transformer.transform(x_max, y_min),
            transformer.transform(x_max, y_max),
        ]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        return (min(lons), min(lats), max(lons), max(lats))
    except Exception as e:
        print(f"    Warning: CRS transform failed ({crs}): {e}")
        return None


def project_ring_to_window_crs(ring_wgs84: list, target_crs: str) -> list:
    """Project a WGS84 coordinate ring to the window's CRS."""
    if target_crs == "EPSG:4326":
        return ring_wgs84

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    projected = []
    for lon, lat in ring_wgs84:
        x, y = transformer.transform(lon, lat)
        projected.append([x, y])
    return projected


def process_window(
    window_dir: Path,
    osm_features: list,
    grid: SpatialGrid,
) -> str:
    """
    Process a single window.  Returns one of:
    - "osm"      : OSM polygons were rasterized
    - "eia"      : kept existing EIA labels (no OSM overlap)
    - "skip"     : window skipped (missing metadata or label file)
    """
    meta = load_window_metadata(window_dir)
    if meta is None:
        return "skip"

    label_path = window_dir / LABEL_SUBPATH
    if not label_path.exists():
        return "skip"

    # Get window bbox in WGS84 for fast overlap filtering
    win_bbox = get_window_bbox_wgs84(meta)
    if win_bbox is None:
        return "skip"

    # Use spatial grid to get candidates, then verify bbox overlap
    candidate_idxs = grid.query(win_bbox)
    overlapping = []
    for idx in candidate_idxs:
        osm_f = osm_features[idx]
        if bboxes_overlap(win_bbox, osm_f["bbox_wgs84"]):
            overlapping.append(osm_f)

    if not overlapping:
        return "eia"

    # Read existing label raster for dimensions
    try:
        existing = tifffile.imread(str(label_path))
    except Exception as e:
        print(f"    Warning: could not read {label_path}: {e}")
        return "skip"

    height, width = existing.shape[:2] if existing.ndim >= 2 else (RASTER_H, RASTER_W)

    # Window bounds in projected CRS (convert from pixel space)
    crs_bounds = pixel_bounds_to_crs(meta)
    if crs_bounds is None:
        return "skip"
    x_min, y_min, x_max, y_max = crs_bounds
    target_crs = meta.get("projection", {}).get("crs", "EPSG:4326")

    # Create new raster -- start from existing so we preserve EIA points
    # in areas not covered by OSM polygons
    new_raster = existing.copy()
    if new_raster.ndim > 2:
        new_raster = new_raster[:, :, 0]  # take first band
    new_raster = new_raster.astype(np.uint8)

    any_pixels_burned = False

    for osm_f in overlapping:
        props = osm_f["feature"]["properties"]
        class_id = props["class_id"]
        ring_wgs84 = osm_f["ring_wgs84"]

        try:
            ring_proj = project_ring_to_window_crs(ring_wgs84, target_crs)
        except Exception:
            continue

        # Check if any part of the projected ring actually falls within the window
        ring_bbox = bbox_from_ring(ring_proj)
        win_proj_bbox = (x_min, y_min, x_max, y_max)
        if not bboxes_overlap(ring_bbox, win_proj_bbox):
            continue

        old_sum = new_raster.sum()
        rasterize_polygon_into(
            new_raster, ring_proj, class_id, x_min, y_min, x_max, y_max, width, height
        )
        if new_raster.sum() != old_sum:
            any_pixels_burned = True

    if not any_pixels_burned:
        return "eia"

    # Back up original label
    backup_path = label_path.with_suffix(".tif.eia_backup")
    if not backup_path.exists():
        try:
            shutil.copy2(str(label_path), str(backup_path))
        except Exception:
            pass

    # Write updated label
    try:
        tifffile.imwrite(str(label_path), new_raster)
    except Exception as e:
        print(f"    Warning: could not write {label_path}: {e}")
        return "skip"

    return "osm"


def main():
    print("=" * 60)
    print("Create Labels from OSM Polygons (v2 — spatial indexed)")
    print("=" * 60)
    print()

    if not GEOJSON_FILE.exists():
        print(f"ERROR: GeoJSON file not found: {GEOJSON_FILE}")
        print("Run download_osm_energy_polygons.py first.")
        sys.exit(1)

    if not WINDOWS_DIR.exists():
        print(f"ERROR: Windows directory not found: {WINDOWS_DIR}")
        print("Make sure the rslearn dataset has been prepared.")
        sys.exit(1)

    # Load OSM features with spatial index
    osm_features, grid = load_osm_features(GEOJSON_FILE)
    if not osm_features:
        print("No OSM polygon features found. Exiting.")
        sys.exit(1)

    # Find all window directories
    window_dirs = sorted([
        d for d in WINDOWS_DIR.iterdir()
        if d.is_dir()
    ])
    total = len(window_dirs)
    print(f"Found {total} windows in {WINDOWS_DIR}")
    print()

    if total == 0:
        print("No windows to process.")
        sys.exit(0)

    # Process windows
    stats = {"osm": 0, "eia": 0, "skip": 0}
    class_counts = defaultdict(int)  # track which classes got burned
    start_time = time.time()

    for i, wdir in enumerate(window_dirs, 1):
        result = process_window(wdir, osm_features, grid)
        stats[result] += 1

        # Progress
        if i % 50 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"  [{i:5d}/{total}] "
                f"OSM={stats['osm']:4d}  EIA={stats['eia']:4d}  "
                f"skip={stats['skip']:3d}  "
                f"({rate:.1f} win/s, ETA {eta:.0f}s)"
            )

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Windows with OSM polygon labels: {stats['osm']:6d}")
    print(f"  Windows kept EIA point labels:   {stats['eia']:6d}")
    print(f"  Windows skipped (errors/missing): {stats['skip']:5d}")
    print(f"  Total processed:                 {total:6d}")
    print(f"  Time elapsed:                    {elapsed:.1f}s")
    print()

    if stats["osm"] > 0:
        pct = stats["osm"] / (stats["osm"] + stats["eia"]) * 100
        print(
            f"  {pct:.1f}% of valid windows upgraded from point to polygon labels."
        )
    print()
    print("Original EIA labels backed up as geotiff.tif.eia_backup")
    print("Done.")


if __name__ == "__main__":
    main()
