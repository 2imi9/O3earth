#!/usr/bin/env python3
"""
Download OpenStreetMap power generator/plant polygons for the entire US.

Uses the Overpass API to query for power=generator and power=plant features,
filters by energy type, and saves the results as a GeoJSON file.

Dependencies: requests (standard), json (stdlib), time (stdlib)
"""

import json
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 180  # seconds per query (Overpass server-side limit)
REQUEST_TIMEOUT = 300   # requests library timeout (client-side)
DELAY_BETWEEN_QUERIES = 12  # seconds between Overpass calls (rate limit)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "osm_energy_polygons.geojson"

# Class mapping: class_id -> (class_name, list of OSM source tag values)
CLASS_MAP = {
    1:  ("solar",       ["solar"]),
    2:  ("wind",        ["wind"]),
    3:  ("gas",         ["gas", "natural_gas"]),
    4:  ("coal",        ["coal"]),
    5:  ("nuclear",     ["nuclear"]),
    6:  ("hydro",       ["hydro", "hydro;solar", "water"]),
    7:  ("oil",         ["oil"]),
    8:  ("biomass",     ["biomass", "biogas", "waste"]),
    9:  ("geothermal",  ["geothermal"]),
    10: ("storage",     ["battery", "battery_storage"]),
}

# Reverse lookup: osm tag value -> (class_id, class_name)
TAG_TO_CLASS = {}
for cid, (cname, tags) in CLASS_MAP.items():
    for tag in tags:
        TAG_TO_CLASS[tag.lower()] = (cid, cname)

# US split into regional bounding boxes (south, west, north, east)
# Kept large enough to cover all territory but small enough to avoid timeouts.
US_REGIONS = [
    # Pacific Northwest
    ("pacific_nw",       42.0, -125.0, 49.0, -116.5),
    # California
    ("california_n",     37.0, -124.5, 42.0, -119.0),
    ("california_s",     32.5, -121.0, 37.0, -114.5),
    # Mountain West
    ("mountain_nw",      42.0, -116.5, 49.0, -109.0),
    ("mountain_sw",      37.0, -116.5, 42.0, -109.0),
    ("mountain_south",   31.0, -114.5, 37.0, -109.0),
    # Great Plains North
    ("plains_nw",        42.0, -109.0, 49.0, -100.0),
    ("plains_sw",        37.0, -109.0, 42.0, -100.0),
    # Great Plains South
    ("plains_south",     31.0, -109.0, 37.0, -100.0),
    ("texas_w",          25.5, -107.0, 31.0, -100.0),
    # Central North
    ("central_n",        42.0, -100.0, 49.0, -92.0),
    ("central_s",        37.0, -100.0, 42.0, -92.0),
    # Central South
    ("central_south_w",  31.0, -100.0, 37.0, -92.0),
    ("texas_e",          25.5, -100.0, 31.0, -92.0),
    # Upper Midwest
    ("midwest_n",        42.0, -92.0, 49.0, -84.0),
    ("midwest_s",        37.0, -92.0, 42.0, -84.0),
    # Lower Mississippi / Gulf
    ("gulf_w",           25.5, -92.0, 31.0, -84.0),
    ("gulf_central",     31.0, -92.0, 37.0, -84.0),
    # Great Lakes / Ohio Valley
    ("greatlakes_w",     42.0, -84.0, 49.0, -76.0),
    ("ohio_valley",      37.0, -84.0, 42.0, -76.0),
    # Southeast
    ("southeast_n",      31.0, -84.0, 37.0, -76.0),
    ("southeast_s",      25.0, -84.0, 31.0, -76.0),
    # Northeast
    ("northeast_n",      42.0, -76.0, 47.5, -66.5),
    ("northeast_s",      37.0, -76.0, 42.0, -66.5),
    # Mid-Atlantic coast
    ("midatlantic",      34.0, -76.0, 37.0, -66.5),
    # Alaska (large but sparse)
    ("alaska_se",        54.0, -141.0, 62.0, -130.0),
    ("alaska_s",         54.0, -170.0, 62.0, -141.0),
    ("alaska_n",         62.0, -170.0, 72.0, -141.0),
    # Hawaii
    ("hawaii",           18.5, -161.0, 22.5, -154.5),
]


def build_overpass_query(south: float, west: float, north: float, east: float) -> str:
    """Build an Overpass QL query for power generators and plants in a bbox."""
    bbox = f"{south},{west},{north},{east}"
    # We request ways and relations (polygons). Nodes are points -- we skip
    # them since we want polygon footprints only.  We also grab nodes that
    # are part of the ways so Overpass can reconstruct the geometry.
    query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["power"="generator"]({bbox});
  way["power"="plant"]({bbox});
  relation["power"="generator"]({bbox});
  relation["power"="plant"]({bbox});
);
out body;
>;
out skel qt;
"""
    return query


def classify_element(tags: dict) -> tuple:
    """Return (class_id, class_name, source_tag) from OSM tags, or None."""
    # Try multiple tag keys in priority order
    for key in (
        "generator:source",
        "plant:source",
        "generator:type",
        "plant:type",
        "source",
    ):
        val = tags.get(key, "").lower().strip()
        if val in TAG_TO_CLASS:
            cid, cname = TAG_TO_CLASS[val]
            return cid, cname, f"{key}={val}"
        # Handle semicolon-separated multi-values (e.g. "hydro;solar")
        for part in val.split(";"):
            part = part.strip()
            if part in TAG_TO_CLASS:
                cid, cname = TAG_TO_CLASS[part]
                return cid, cname, f"{key}={part}"
    return None


def elements_to_features(data: dict) -> list:
    """
    Convert Overpass JSON response into GeoJSON features.

    Overpass returns nodes, ways, and relations separately.  We need to
    reconstruct way geometries from their node references.
    """
    # Build node lookup
    nodes = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])

    features = []

    for el in data.get("elements", []):
        tags = el.get("tags", {})

        # Only process ways/relations that have power tags
        if el["type"] == "way" and "power" in tags:
            classification = classify_element(tags)
            if classification is None:
                continue
            class_id, class_name, source_tag = classification

            # Build coordinate ring from node refs
            coords = []
            for nid in el.get("nodes", []):
                if nid in nodes:
                    coords.append(list(nodes[nid]))
            if len(coords) < 3:
                continue

            # Close the ring if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "class_id": class_id,
                    "class_name": class_name,
                    "osm_id": f"way/{el['id']}",
                    "source_tag": source_tag,
                },
            }
            features.append(feature)

        elif el["type"] == "relation" and "power" in tags:
            classification = classify_element(tags)
            if classification is None:
                continue
            class_id, class_name, source_tag = classification

            # For relations, try to build multipolygon from outer members
            outer_rings = []
            for member in el.get("members", []):
                if member.get("role") in ("outer", "") and member["type"] == "way":
                    # We don't have the way geometry inline for relations
                    # in the skeleton output; we'd need a second pass.
                    # For simplicity, skip relation geometry reconstruction
                    # (most energy plants are mapped as ways, not relations).
                    pass

            # Even if we can't reconstruct the full geometry, record the
            # relation if it has a center (Overpass "out center" would give
            # this, but we use "out body" + skel).  We'll skip relations
            # without reconstructable geometry.
            # Future improvement: use out center for relations.
            pass

    return features


def query_region(
    name: str,
    south: float,
    west: float,
    north: float,
    east: float,
    session: requests.Session,
) -> list:
    """Query one region and return GeoJSON features."""
    query = build_overpass_query(south, west, north, east)
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = 60 * attempt
                print(f"  Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            if resp.status_code == 504:
                print(f"  Gateway timeout (attempt {attempt}/{max_retries}).")
                time.sleep(30)
                continue
            resp.raise_for_status()
            data = resp.json()
            features = elements_to_features(data)
            return features

        except requests.exceptions.Timeout:
            print(f"  Client timeout (attempt {attempt}/{max_retries}).")
            time.sleep(30)
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(30)
        except Exception as e:
            print(f"  Unexpected error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(15)

    print(f"  FAILED after {max_retries} attempts for region {name}.")
    return []


def main():
    print("=" * 60)
    print("OSM Energy Polygon Downloader")
    print("=" * 60)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Regions to query: {len(US_REGIONS)}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_features = []
    seen_osm_ids = set()
    session = requests.Session()
    session.headers.update({"User-Agent": "OpenEnergy-Engine/1.0 (research)"})

    for i, (name, south, west, north, east) in enumerate(US_REGIONS, 1):
        pct = (i - 1) / len(US_REGIONS) * 100
        print(
            f"[{i:2d}/{len(US_REGIONS)}] ({pct:5.1f}%) Querying {name} "
            f"({south},{west} -> {north},{east}) ..."
        )

        features = query_region(name, south, west, north, east, session)

        new_count = 0
        for f in features:
            osm_id = f["properties"]["osm_id"]
            if osm_id not in seen_osm_ids:
                seen_osm_ids.add(osm_id)
                all_features.append(f)
                new_count += 1

        print(
            f"  -> {len(features)} features ({new_count} new, "
            f"{len(features) - new_count} duplicates skipped)"
        )

        # Rate-limit delay (skip after last query)
        if i < len(US_REGIONS):
            print(f"  Waiting {DELAY_BETWEEN_QUERIES}s for rate limit...")
            time.sleep(DELAY_BETWEEN_QUERIES)

    # Summary by class
    print()
    print("=" * 60)
    print(f"Total unique features: {len(all_features)}")
    print()
    class_counts = {}
    for f in all_features:
        cname = f["properties"]["class_name"]
        class_counts[cname] = class_counts.get(cname, 0) + 1
    for cid, (cname, _) in sorted(CLASS_MAP.items()):
        count = class_counts.get(cname, 0)
        print(f"  {cid:2d} {cname:12s}: {count:6d}")
    print()

    # Write GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    print(f"Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
