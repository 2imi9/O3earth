#!/usr/bin/env python3
"""Download all US renewable energy plants from EIA API v2 with retry logic."""

import os, requests, pandas as pd, time, json, sys
from pathlib import Path

EIA_API_KEY = os.environ.get("EIA_API_KEY")
if not EIA_API_KEY:
    raise RuntimeError("Set EIA_API_KEY environment variable. Get one at https://www.eia.gov/opendata/register.php")
EIA_BASE = "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"
FUEL_MAP = {"SUN": "solar", "WND": "wind", "WAT": "hydro", "GEO": "geothermal"}
OUTPUT = Path("data/eia_plants.parquet")

def fetch_fuel(fuel, existing_offset=0, existing_plants=None):
    plants = existing_plants or []
    offset = existing_offset
    retries = 0
    while True:
        try:
            resp = requests.get(EIA_BASE, params={
                "api_key": EIA_API_KEY,
                "frequency": "monthly",
                "data[0]": "latitude", "data[1]": "longitude",
                "data[2]": "nameplate-capacity-mw", "data[3]": "operating-year-month",
                "facets[energy_source_code][]": fuel,
                "facets[status][]": "OP",
                "sort[0][column]": "plantid", "sort[0][direction]": "asc",
                "offset": offset, "length": 5000,
            }, timeout=120)
            resp.raise_for_status()
            records = resp.json().get("response", {}).get("data", [])
            if not records:
                break
            for r in records:
                lat, lon = r.get("latitude"), r.get("longitude")
                if lat and lon:
                    plants.append({
                        "lat": float(lat), "lon": float(lon),
                        "energy_type": FUEL_MAP[fuel], "source": "eia_860",
                        "name": r.get("plantName", ""),
                        "capacity_mw": float(r.get("nameplate-capacity-mw", 0) or 0),
                        "country_code": "US",
                        "operating_year": str(r.get("operating-year-month", ""))[:4],
                    })
            offset += 5000
            retries = 0
            print(f"  {fuel}: {len(plants)} records (offset={offset})", flush=True)
            if len(records) < 5000:
                break
            time.sleep(0.5)
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f"  {fuel}: giving up after 5 retries at offset={offset}", flush=True)
                break
            print(f"  {fuel}: retry {retries} at offset={offset}: {e}", flush=True)
            time.sleep(5 * retries)
    return plants

all_plants = []
for fuel in ["SUN", "WND", "WAT", "GEO"]:
    print(f"\n=== Downloading {fuel} ({FUEL_MAP[fuel]}) ===", flush=True)
    plants = fetch_fuel(fuel)
    all_plants.extend(plants)

df = pd.DataFrame(all_plants)
df = df.drop_duplicates(subset=["lat", "lon", "energy_type"], keep="first")
print(f"\nTotal (deduplicated): {len(df)}")
print(df["energy_type"].value_counts().to_string())

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
