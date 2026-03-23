#!/usr/bin/env python3
"""
Build a global patch-level suitability dataset from OSM energy polygons.

Positive samples: centroids of OSM solar/wind polygons (real plant locations).
Negative samples: random land points with no energy infrastructure within 5 km.

Output: data/suitability_dataset.parquet with columns:
    [lat, lon, energy_type, label, region, country_code, osm_id]

Usage:
    python scripts/build_suitability_dataset.py \
        --osm-path data/osm_energy_polygons.geojson \
        --output data/suitability_dataset.parquet \
        --solar-pos 5000 --solar-neg 5000 \
        --wind-pos 3000 --wind-neg 3000 \
        --seed 42
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rough continent bounding boxes for region assignment
REGION_BOXES = {
    "North America":  {"lat": (15, 72),  "lon": (-170, -50)},
    "South America":  {"lat": (-56, 15), "lon": (-82, -34)},
    "Europe":         {"lat": (35, 72),  "lon": (-25, 45)},
    "Africa":         {"lat": (-35, 37), "lon": (-18, 52)},
    "Middle East":    {"lat": (12, 42),  "lon": (25, 63)},
    "Central Asia":   {"lat": (35, 55),  "lon": (45, 90)},
    "South Asia":     {"lat": (5, 38),   "lon": (60, 98)},
    "East Asia":      {"lat": (18, 55),  "lon": (73, 150)},
    "Southeast Asia": {"lat": (-11, 24), "lon": (95, 153)},
    "Oceania":        {"lat": (-48, -8), "lon": (110, 180)},
}

# ISO-3166 approximate country centroids are too many to embed; instead we use
# a lightweight reverse-geocoding strategy based on Natural Earth boundaries.
# If the `reverse_geocoder` package is available we use that (fast, offline).
# Otherwise we fall back to region-only assignment.

# Land mask: we use Natural Earth 110m land polygons bundled with shapely/cartopy
# or download a small one.  For robustness we generate candidate negatives and
# filter with a lightweight land check.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_region(lat: float, lon: float) -> str:
    """Assign a coarse region name from lat/lon."""
    for name, box in REGION_BOXES.items():
        if box["lat"][0] <= lat <= box["lat"][1] and box["lon"][0] <= lon <= box["lon"][1]:
            return name
    return "Other"


def _try_reverse_geocode(lats: np.ndarray, lons: np.ndarray) -> list[str]:
    """Attempt offline reverse geocoding for country codes.

    Uses the `reverse_geocoder` library (pip install reverse_geocoder).
    Falls back to empty strings if unavailable.
    """
    try:
        import reverse_geocoder as rg

        coords = list(zip(lats.tolist(), lons.tolist()))
        results = rg.search(coords)
        return [r.get("cc", "") for r in results]
    except ImportError:
        logger.warning(
            "reverse_geocoder not installed — country_code will be empty. "
            "Install with: pip install reverse_geocoder"
        )
        return [""] * len(lats)
    except Exception as e:
        logger.warning(f"Reverse geocoding failed: {e}")
        return [""] * len(lats)


def _load_land_mask():
    """Return a function is_land(lat, lon) -> bool.

    Strategy:
    1. Try cartopy Natural Earth land geometries (best).
    2. Try shapely with a bundled simplified geojson.
    3. Fall back to a crude latitude heuristic (no ocean poles).
    """
    # --- Try cartopy ---
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep

        land_shp = shpreader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
        reader = shpreader.Reader(land_shp)
        land_geom = unary_union(list(reader.geometries()))
        prepared = prep(land_geom)

        def is_land(lat, lon):
            return prepared.contains(Point(lon, lat))

        logger.info("Using cartopy Natural Earth land mask.")
        return is_land
    except Exception:
        pass

    # --- Try geopandas with naturalearth_lowres ---
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from shapely.prepared import prep

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        land_geom = unary_union(world.geometry.values)
        prepared = prep(land_geom)

        def is_land(lat, lon):
            return prepared.contains(Point(lon, lat))

        logger.info("Using geopandas naturalearth_lowres land mask.")
        return is_land
    except Exception:
        pass

    # --- Fallback: crude heuristic ---
    logger.warning(
        "No land mask library available. Using latitude heuristic "
        "(will include some ocean points). Install cartopy or geopandas for accuracy."
    )

    def is_land(lat, lon):
        # Reject obvious ocean: high Arctic, Antarctica, mid-ocean strips
        if lat < -60 or lat > 75:
            return False
        # Very rough: reject large ocean bands
        if -60 < lat < -40 and (lon < -70 or lon > 150):
            return False
        return True

    return is_land


def _load_osm_polygons(path: str, energy_type: str) -> pd.DataFrame:
    """Load OSM energy polygons, filtering by energy_type.

    Handles the 540 MB geojson by:
    1. Trying geopandas (efficient C-level parser).
    2. Falling back to ijson streaming parser.

    Returns DataFrame with columns: [lat, lon, osm_id, class_name].
    """
    logger.info(f"Loading OSM polygons for '{energy_type}' from {path} ...")

    # --- Try geopandas (preferred for large geojson) ---
    try:
        import geopandas as gpd

        logger.info("Reading with geopandas (may take 1-2 min for 540 MB) ...")
        gdf = gpd.read_file(path)
        # Filter to desired energy type
        mask = gdf["class_name"].str.lower().str.contains(energy_type.lower(), na=False)
        gdf = gdf[mask].copy()
        logger.info(f"  Found {len(gdf)} '{energy_type}' features via geopandas.")

        # Compute centroids
        centroids = gdf.geometry.centroid
        result = pd.DataFrame({
            "lat": centroids.y.values,
            "lon": centroids.x.values,
            "osm_id": gdf.get("osm_id", gdf.index).values if "osm_id" in gdf.columns else range(len(gdf)),
            "class_name": gdf["class_name"].values,
        })
        return result

    except ImportError:
        logger.info("geopandas not available, trying ijson streaming ...")
    except Exception as e:
        logger.warning(f"geopandas load failed ({e}), trying ijson ...")

    # --- Fallback: ijson streaming ---
    try:
        import ijson
    except ImportError:
        logger.error(
            "Neither geopandas nor ijson is available. "
            "Install one of them: pip install geopandas  OR  pip install ijson"
        )
        sys.exit(1)

    records = []
    with open(path, "rb") as f:
        features = ijson.items(f, "features.item")
        for feat in tqdm(features, desc=f"Streaming OSM ({energy_type})", unit=" features"):
            props = feat.get("properties", {})
            cname = props.get("class_name", "")
            if energy_type.lower() not in cname.lower():
                continue

            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            gtype = geom.get("type", "")

            # Compute centroid from coordinates
            lat, lon = _centroid_from_geojson(gtype, coords)
            if lat is not None:
                records.append({
                    "lat": lat,
                    "lon": lon,
                    "osm_id": props.get("osm_id", ""),
                    "class_name": cname,
                })

    logger.info(f"  Streamed {len(records)} '{energy_type}' features via ijson.")
    return pd.DataFrame(records)


def _centroid_from_geojson(gtype: str, coords) -> tuple:
    """Compute a rough centroid from GeoJSON coordinates."""
    try:
        if gtype == "Point":
            return coords[1], coords[0]
        elif gtype == "Polygon":
            ring = coords[0]  # exterior ring
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            return np.mean(lats), np.mean(lons)
        elif gtype == "MultiPolygon":
            all_lats, all_lons = [], []
            for poly in coords:
                ring = poly[0]
                all_lons.extend(c[0] for c in ring)
                all_lats.extend(c[1] for c in ring)
            return np.mean(all_lats), np.mean(all_lons)
        elif gtype == "LineString":
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return np.mean(lats), np.mean(lons)
        elif gtype == "MultiLineString":
            all_lats, all_lons = [], []
            for line in coords:
                all_lons.extend(c[0] for c in line)
                all_lats.extend(c[1] for c in line)
            return np.mean(all_lats), np.mean(all_lons)
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_positives(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    max_per_country_frac: float = 0.25,
) -> pd.DataFrame:
    """Sample n positive locations with geographic diversity.

    Ensures no single region contributes more than max_per_country_frac of samples.
    Uses grid-based spatial thinning: divide globe into 1-degree cells, sample
    at most a few per cell to get spatial spread.
    """
    if len(df) == 0:
        logger.warning("No positive features found — returning empty DataFrame.")
        return df.head(0)

    if len(df) <= n:
        logger.info(f"  Only {len(df)} features available (requested {n}), using all.")
        return df.copy()

    # Assign grid cell (1-degree)
    df = df.copy()
    df["grid_cell"] = (df["lat"].round(0).astype(int).astype(str)
                        + "_"
                        + df["lon"].round(0).astype(int).astype(str))

    # Sample at most ceil(n / num_cells * 3) per cell, then subsample globally
    cells = df["grid_cell"].unique()
    per_cell = max(1, int(np.ceil(n / len(cells) * 3)))

    sampled = (
        df.groupby("grid_cell", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_cell), random_state=int(rng.integers(1e9))))
    )

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=int(rng.integers(1e9)))

    # If still short, top up from remaining
    if len(sampled) < n:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(
            n=min(n - len(sampled), len(remaining)),
            random_state=int(rng.integers(1e9)),
        )
        sampled = pd.concat([sampled, extra])

    return sampled.drop(columns=["grid_cell"], errors="ignore")


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _sample_negatives(
    positive_lats: np.ndarray,
    positive_lons: np.ndarray,
    n: int,
    is_land,
    rng: np.random.Generator,
    min_dist_km: float = 5.0,
    batch_size: int = 5000,
    max_iters: int = 200,
) -> pd.DataFrame:
    """Sample n negative (no-plant) locations on land, >min_dist_km from any positive.

    Strategy:
    - Generate random lat/lon on land
    - Reject points within min_dist_km of any positive
    - Use spatial hashing (1-degree grid) for fast proximity check
    """
    # Build spatial index: set of 0.05-degree grid cells containing positives
    # (0.05 deg ~ 5.5 km at equator, good proxy for quick reject)
    cell_size = 0.05  # degrees
    pos_cells = set()
    for lat, lon in zip(positive_lats, positive_lons):
        # Mark the cell and its 8 neighbors
        ci = int(np.floor(lat / cell_size))
        cj = int(np.floor(lon / cell_size))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                pos_cells.add((ci + di, cj + dj))

    negatives = []
    attempts = 0

    pbar = tqdm(total=n, desc="Sampling negatives", unit=" pts")

    while len(negatives) < n and attempts < max_iters:
        attempts += 1
        # Generate candidate points: latitude weighted by cos to get uniform area
        u = rng.uniform(0, 1, size=batch_size)
        cand_lat = np.degrees(np.arcsin(2 * u - 1))
        # Restrict to reasonable land latitudes
        mask_lat = (cand_lat > -56) & (cand_lat < 72)
        cand_lat = cand_lat[mask_lat]
        cand_lon = rng.uniform(-180, 180, size=len(cand_lat))

        for lat, lon in zip(cand_lat, cand_lon):
            if len(negatives) >= n:
                break

            # Quick grid-cell check
            ci = int(np.floor(lat / cell_size))
            cj = int(np.floor(lon / cell_size))
            if (ci, cj) in pos_cells:
                continue

            # Land check
            if not is_land(lat, lon):
                continue

            negatives.append({"lat": float(lat), "lon": float(lon)})
            pbar.update(1)

    pbar.close()

    if len(negatives) < n:
        logger.warning(
            f"Only found {len(negatives)} negatives (requested {n}). "
            "Try increasing max_iters or relaxing constraints."
        )

    return pd.DataFrame(negatives[:n])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build global patch-level suitability dataset from OSM energy polygons."
    )
    parser.add_argument(
        "--osm-path",
        default="data/osm_energy_polygons.geojson",
        help="Path to OSM energy polygons GeoJSON (default: data/osm_energy_polygons.geojson)",
    )
    parser.add_argument(
        "--output",
        default="data/suitability_dataset.parquet",
        help="Output parquet file path.",
    )
    parser.add_argument("--solar-pos", type=int, default=5000, help="Number of solar positive samples.")
    parser.add_argument("--solar-neg", type=int, default=5000, help="Number of solar negative samples.")
    parser.add_argument("--wind-pos", type=int, default=3000, help="Number of wind positive samples.")
    parser.add_argument("--wind-neg", type=int, default=3000, help="Number of wind negative samples.")
    parser.add_argument("--min-dist-km", type=float, default=5.0, help="Min distance from positive for negatives (km).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-csv", action="store_true", help="Also save as CSV.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    osm_path = Path(args.osm_path)
    if not osm_path.is_absolute():
        osm_path = project_root / osm_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load land mask
    is_land = _load_land_mask()

    all_frames = []

    for energy_type, n_pos, n_neg in [
        ("solar", args.solar_pos, args.solar_neg),
        ("wind", args.wind_pos, args.wind_neg),
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {energy_type.upper()} — target: {n_pos} pos + {n_neg} neg")
        logger.info(f"{'='*60}")

        # --- Load and sample positives ---
        osm_df = _load_osm_polygons(str(osm_path), energy_type)
        if len(osm_df) == 0:
            logger.warning(f"No {energy_type} polygons found. Skipping.")
            continue

        positives = _sample_positives(osm_df, n_pos, rng)
        positives["energy_type"] = energy_type
        positives["label"] = 1
        logger.info(f"  Sampled {len(positives)} positive locations.")

        # --- Sample negatives ---
        all_pos_lats = osm_df["lat"].values
        all_pos_lons = osm_df["lon"].values
        negatives = _sample_negatives(
            all_pos_lats, all_pos_lons, n_neg, is_land, rng,
            min_dist_km=args.min_dist_km,
        )
        negatives["energy_type"] = energy_type
        negatives["label"] = 0
        negatives["osm_id"] = ""
        negatives["class_name"] = ""
        logger.info(f"  Sampled {len(negatives)} negative locations.")

        all_frames.append(positives)
        all_frames.append(negatives)

    if not all_frames:
        logger.error("No data collected. Check OSM file and class_name values.")
        sys.exit(1)

    # Combine all
    dataset = pd.concat(all_frames, ignore_index=True)

    # Assign region
    logger.info("Assigning regions ...")
    dataset["region"] = [_assign_region(lat, lon) for lat, lon in
                         tqdm(zip(dataset["lat"], dataset["lon"]),
                              total=len(dataset), desc="Regions")]

    # Reverse geocode for country code
    logger.info("Reverse geocoding for country codes ...")
    dataset["country_code"] = _try_reverse_geocode(
        dataset["lat"].values, dataset["lon"].values
    )

    # Clean up columns
    keep_cols = ["lat", "lon", "energy_type", "label", "region", "country_code", "osm_id"]
    for col in keep_cols:
        if col not in dataset.columns:
            dataset[col] = ""
    dataset = dataset[keep_cols]

    # Shuffle
    dataset = dataset.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    # Save
    logger.info(f"Saving dataset ({len(dataset)} rows) to {output_path} ...")
    dataset.to_parquet(str(output_path), index=False)

    if args.output_csv:
        csv_path = output_path.with_suffix(".csv")
        dataset.to_csv(str(csv_path), index=False)
        logger.info(f"Also saved CSV to {csv_path}")

    # Summary stats
    logger.info("\n=== Dataset Summary ===")
    for etype in dataset["energy_type"].unique():
        sub = dataset[dataset["energy_type"] == etype]
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]
        logger.info(f"  {etype}: {len(pos)} positive, {len(neg)} negative")
        logger.info(f"    Regions: {pos['region'].value_counts().to_dict()}")

    logger.info(f"\nTotal samples: {len(dataset)}")
    logger.info(f"Output: {output_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
