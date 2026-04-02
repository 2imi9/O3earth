#!/usr/bin/env python3
"""
download_global_energy_locations.py — Download global energy plant locations from multiple sources:

1. EIA 860 (US plants — all fuel types with lat/lon)
2. OSM Overpass API (global solar/wind/hydro/geothermal)
3. Existing OSM geojson (already downloaded, US-heavy)

Combines into a single global dataset for suitability training.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EIA_API_KEY = os.environ.get("EIA_API_KEY", "r84SSU7MzfwG3lbExInY26kc5Ek5fmculBx7Kt1J")
EIA_BASE = "https://api.eia.gov/v2"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Energy type mapping
EIA_FUEL_MAP = {
    "SUN": "solar", "WND": "wind", "WAT": "hydro",
    "GEO": "geothermal", "NG": "gas", "COL": "coal",
    "NUC": "nuclear", "DFO": "oil", "RFO": "oil",
    "WDS": "biomass", "BIT": "coal", "SUB": "coal",
    "AB": "biomass", "WH": "other", "OBG": "biomass",
    "LFG": "biomass", "MWH": "storage",
}


def download_eia_plants():
    """Download all US power plants from EIA 860 via API v2."""
    log.info("Downloading EIA 860 plant data...")

    all_plants = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "annual",
            "data[0]": "nameplate-capacity-mw",
            "facets[sectorDescription][]": "Electric Utility",
            "sort[0][column]": "plantCode",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size,
        }

        try:
            resp = requests.get(f"{EIA_BASE}/electricity/operating-generator-capacity/data/", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("response", {}).get("data", [])
            if not records:
                break

            for r in records:
                lat = r.get("latitude")
                lon = r.get("longitude")
                fuel = r.get("energy_source_code") or r.get("energySourceCode", "")
                if lat and lon and fuel:
                    try:
                        all_plants.append({
                            "lat": float(lat),
                            "lon": float(lon),
                            "energy_type": EIA_FUEL_MAP.get(fuel, "other"),
                            "source": "eia_860",
                            "name": r.get("plantName", ""),
                            "capacity_mw": float(r.get("nameplate-capacity-mw", 0) or 0),
                            "country_code": "US",
                            "fuel_code": fuel,
                            "operating_year": r.get("operatingYear", ""),
                        })
                    except (ValueError, TypeError):
                        continue

            offset += page_size
            log.info(f"  EIA: fetched {len(all_plants)} plants so far (offset={offset})")

            if len(records) < page_size:
                break

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            log.error(f"  EIA API error at offset {offset}: {e}")
            break

    df = pd.DataFrame(all_plants)
    log.info(f"  EIA total: {len(df)} plants")
    if len(df) > 0:
        log.info(f"  By type: {df['energy_type'].value_counts().to_dict()}")
    return df


def query_overpass(query, description=""):
    """Run an Overpass API query."""
    log.info(f"  Overpass query: {description}")
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
        resp.raise_for_status()
        return resp.json().get("elements", [])
    except Exception as e:
        log.error(f"  Overpass error: {e}")
        return []


def download_osm_global_solar():
    """Download global solar farm locations from OSM via Overpass API."""
    log.info("Downloading global solar farms from OSM Overpass...")

    # Query large solar farms (>1MW typically mapped as ways/relations)
    # Split by continent to avoid timeout
    regions = {
        "Europe": (35, -25, 72, 45),
        "Asia": (0, 45, 60, 180),
        "Africa": (-35, -20, 37, 55),
        "South America": (-55, -82, 15, -34),
        "Oceania": (-47, 110, -10, 180),
        "East Asia": (18, 73, 55, 145),
    }

    all_solar = []
    for region_name, (s, w, n, e) in regions.items():
        query = f"""
        [out:json][timeout:120];
        (
          way["plant:source"="solar"]({s},{w},{n},{e});
          way["generator:source"="solar"]["generator:output:electricity"~"MW"]({s},{w},{n},{e});
          relation["plant:source"="solar"]({s},{w},{n},{e});
        );
        out center;
        """
        elements = query_overpass(query, f"Solar in {region_name}")

        for el in elements:
            lat = el.get("center", {}).get("lat") or el.get("lat")
            lon = el.get("center", {}).get("lon") or el.get("lon")
            if lat and lon:
                tags = el.get("tags", {})
                all_solar.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "energy_type": "solar",
                    "source": "osm_overpass",
                    "name": tags.get("name", ""),
                    "capacity_mw": _parse_capacity(tags),
                    "country_code": "",
                    "fuel_code": "SUN",
                    "operating_year": "",
                })

        log.info(f"    {region_name}: {len(elements)} solar facilities")
        time.sleep(2)  # Be nice to Overpass

    return pd.DataFrame(all_solar)


def download_osm_global_wind():
    """Download global wind farm locations from OSM via Overpass API."""
    log.info("Downloading global wind farms from OSM Overpass...")

    regions = {
        "Europe": (35, -25, 72, 45),
        "Asia": (0, 45, 60, 180),
        "Africa": (-35, -20, 37, 55),
        "South America": (-55, -82, 15, -34),
        "Oceania": (-47, 110, -10, 180),
        "East Asia": (18, 73, 55, 145),
        "North America": (15, -170, 72, -50),
    }

    all_wind = []
    for region_name, (s, w, n, e) in regions.items():
        query = f"""
        [out:json][timeout:120];
        (
          node["generator:source"="wind"]({s},{w},{n},{e});
          way["plant:source"="wind"]({s},{w},{n},{e});
          relation["plant:source"="wind"]({s},{w},{n},{e});
        );
        out center;
        """
        elements = query_overpass(query, f"Wind in {region_name}")

        for el in elements:
            lat = el.get("center", {}).get("lat") or el.get("lat")
            lon = el.get("center", {}).get("lon") or el.get("lon")
            if lat and lon:
                tags = el.get("tags", {})
                all_wind.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "energy_type": "wind",
                    "source": "osm_overpass",
                    "name": tags.get("name", ""),
                    "capacity_mw": _parse_capacity(tags),
                    "country_code": "",
                    "fuel_code": "WND",
                    "operating_year": "",
                })

        log.info(f"    {region_name}: {len(elements)} wind facilities")
        time.sleep(2)

    return pd.DataFrame(all_wind)


def download_osm_global_hydro():
    """Download global hydropower locations from OSM."""
    log.info("Downloading global hydro from OSM Overpass...")

    query = """
    [out:json][timeout:120];
    (
      way["plant:source"="hydro"];
      relation["plant:source"="hydro"];
      way["generator:source"="hydro"]["generator:output:electricity"~"MW"];
    );
    out center;
    """
    elements = query_overpass(query, "Global hydro")

    all_hydro = []
    for el in elements:
        lat = el.get("center", {}).get("lat") or el.get("lat")
        lon = el.get("center", {}).get("lon") or el.get("lon")
        if lat and lon:
            tags = el.get("tags", {})
            all_hydro.append({
                "lat": float(lat),
                "lon": float(lon),
                "energy_type": "hydro",
                "source": "osm_overpass",
                "name": tags.get("name", ""),
                "capacity_mw": _parse_capacity(tags),
                "country_code": "",
                "fuel_code": "WAT",
                "operating_year": "",
            })

    return pd.DataFrame(all_hydro)


def download_osm_global_geothermal():
    """Download global geothermal locations from OSM."""
    log.info("Downloading global geothermal from OSM Overpass...")

    query = """
    [out:json][timeout:60];
    (
      node["plant:source"="geothermal"];
      way["plant:source"="geothermal"];
      relation["plant:source"="geothermal"];
      node["generator:source"="geothermal"];
    );
    out center;
    """
    elements = query_overpass(query, "Global geothermal")

    all_geo = []
    for el in elements:
        lat = el.get("center", {}).get("lat") or el.get("lat")
        lon = el.get("center", {}).get("lon") or el.get("lon")
        if lat and lon:
            tags = el.get("tags", {})
            all_geo.append({
                "lat": float(lat),
                "lon": float(lon),
                "energy_type": "geothermal",
                "source": "osm_overpass",
                "name": tags.get("name", ""),
                "capacity_mw": _parse_capacity(tags),
                "country_code": "",
                "fuel_code": "GEO",
                "operating_year": "",
            })

    return pd.DataFrame(all_geo)


def _parse_capacity(tags):
    """Try to parse capacity from OSM tags."""
    for key in ["plant:output:electricity", "generator:output:electricity"]:
        val = tags.get(key, "")
        if val:
            try:
                # "100 MW" or "100MW" or "100"
                num = "".join(c for c in val.split("MW")[0].split("mw")[0] if c.isdigit() or c == ".")
                if num:
                    return float(num)
            except (ValueError, IndexError):
                pass
    return 0.0


def deduplicate(df, distance_threshold_deg=0.005):
    """Remove near-duplicate locations (within ~500m)."""
    if len(df) == 0:
        return df

    log.info(f"  Deduplicating {len(df)} records (threshold={distance_threshold_deg} deg)...")

    # Round to grid and drop duplicates
    df = df.copy()
    df["lat_round"] = (df["lat"] / distance_threshold_deg).round() * distance_threshold_deg
    df["lon_round"] = (df["lon"] / distance_threshold_deg).round() * distance_threshold_deg

    # Keep first occurrence per grid cell per energy type
    df = df.drop_duplicates(subset=["lat_round", "lon_round", "energy_type"], keep="first")
    df = df.drop(columns=["lat_round", "lon_round"])

    log.info(f"  After dedup: {len(df)} records")
    return df


def add_country_codes(df):
    """Add country codes using reverse geocoder."""
    if len(df) == 0:
        return df

    missing = df["country_code"] == ""
    if missing.sum() == 0:
        return df

    try:
        import reverse_geocoder as rg
        coords = list(zip(df.loc[missing, "lat"], df.loc[missing, "lon"]))
        results = rg.search(coords)
        df.loc[missing, "country_code"] = [r["cc"] for r in results]
        log.info(f"  Geocoded {missing.sum()} locations")
    except ImportError:
        log.warning("  reverse_geocoder not installed, skipping country codes")
    except Exception as e:
        log.warning(f"  Geocoding error: {e}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download global energy plant locations")
    parser.add_argument("--output", default="data/global_energy_locations.parquet")
    parser.add_argument("--skip-eia", action="store_true", help="Skip EIA download")
    parser.add_argument("--skip-osm", action="store_true", help="Skip OSM Overpass download")
    parser.add_argument("--energy-types", nargs="+",
                        default=["solar", "wind", "hydro", "geothermal"],
                        help="Energy types to download")
    args = parser.parse_args()

    frames = []

    # 1. EIA 860 (US plants)
    if not args.skip_eia:
        eia_df = download_eia_plants()
        if len(eia_df) > 0:
            frames.append(eia_df)

    # 2. OSM Overpass (global)
    if not args.skip_osm:
        if "solar" in args.energy_types:
            solar_df = download_osm_global_solar()
            if len(solar_df) > 0:
                frames.append(solar_df)

        if "wind" in args.energy_types:
            wind_df = download_osm_global_wind()
            if len(wind_df) > 0:
                frames.append(wind_df)

        if "hydro" in args.energy_types:
            hydro_df = download_osm_global_hydro()
            if len(hydro_df) > 0:
                frames.append(hydro_df)

        if "geothermal" in args.energy_types:
            geo_df = download_osm_global_geothermal()
            if len(geo_df) > 0:
                frames.append(geo_df)

    if not frames:
        log.error("No data downloaded!")
        return

    # Combine
    df = pd.concat(frames, ignore_index=True)
    log.info(f"\nCombined: {len(df)} total records")

    # Deduplicate
    df = deduplicate(df)

    # Add country codes
    df = add_country_codes(df)

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"FINAL DATASET: {len(df)} plant locations")
    log.info(f"{'='*60}")
    log.info(f"By energy type:")
    for et, count in df["energy_type"].value_counts().items():
        log.info(f"  {et}: {count}")
    log.info(f"By source:")
    for src, count in df["source"].value_counts().items():
        log.info(f"  {src}: {count}")
    log.info(f"By country (top 15):")
    for cc, count in df["country_code"].value_counts().head(15).items():
        log.info(f"  {cc}: {count}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"\nSaved to {output_path}")

    # Also save as CSV for easy inspection
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info(f"Also saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
