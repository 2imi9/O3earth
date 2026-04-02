"""
Real-time data fetcher for suitability scoring.

Fetches data from free public APIs to feed the factor engine:
- NASA POWER: solar irradiance, wind speed, temperature, precipitation
- Open-Meteo Flood: river discharge
- USGS Earthquake: seismic activity
- Open-Elevation: terrain elevation/gradient
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds per API call


def fetch_all(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch real-time data from all APIs for a location.

    Returns a dict of kwargs ready to pass to SuitabilityEngine.score().
    Failed API calls are silently skipped (factor falls back to 0.5).
    """
    data: Dict[str, Any] = {}

    # Run all fetchers, catch errors individually
    for name, fetcher in [
        ("nasa_power", _fetch_nasa_power),
        ("elevation", _fetch_elevation),
        ("flood", _fetch_flood_discharge),
        ("earthquake", _fetch_earthquake),
    ]:
        try:
            result = fetcher(lat, lon)
            if result:
                data.update(result)
        except Exception:
            logger.warning("Failed to fetch %s data for (%.4f, %.4f)", name, lat, lon, exc_info=True)

    return data


# ---------------------------------------------------------------------------
# NASA POWER API — solar, wind, temperature, precipitation, cloud cover
# https://power.larc.nasa.gov/docs/services/api/
# ---------------------------------------------------------------------------

def _fetch_nasa_power(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch climatology data from NASA POWER (no auth needed)."""
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,WS10M,T2M,PRECTOTCORR,CLOUD_AMT",
        "community": "RE",  # Renewable Energy
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    props = data.get("properties", {}).get("parameter", {})
    result = {}

    # Solar: ALLSKY_SFC_SW_DWN = GHI in kWh/m2/day (annual average)
    ghi = props.get("ALLSKY_SFC_SW_DWN", {}).get("ANN")
    if ghi is not None and ghi > 0:
        result["ghi_kwh_m2_day"] = ghi

    # Wind: WS10M = wind speed at 10m in m/s (annual average)
    # Scale 10m -> ~80m hub height using power law: v80 = v10 * (80/10)^0.14
    ws10 = props.get("WS10M", {}).get("ANN")
    if ws10 is not None and ws10 > 0:
        result["wind_speed_ms"] = ws10 * (80 / 10) ** 0.14

    # Temperature: T2M = 2m temperature in Celsius (annual average)
    temp = props.get("T2M", {}).get("ANN")
    if temp is not None and temp > -100:
        result["avg_temp_c"] = temp

    # Cloud: CLOUD_AMT = cloud amount percentage (annual average)
    cloud = props.get("CLOUD_AMT", {}).get("ANN")
    if cloud is not None and cloud >= 0:
        result["cloud_fraction"] = cloud / 100.0  # Convert % to fraction

    # Precipitation: PRECTOTCORR in mm/day (annual average)
    precip = props.get("PRECTOTCORR", {}).get("ANN")
    if precip is not None and precip >= 0:
        result["precipitation_mm_day"] = precip

    return result


# ---------------------------------------------------------------------------
# Open-Elevation API — terrain elevation
# https://open-elevation.com/
# ---------------------------------------------------------------------------

def _fetch_elevation(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch elevation and compute local gradient from Open-Elevation API."""
    # Query center + 4 cardinal points ~500m away for gradient
    delta = 0.005  # ~500m at mid-latitudes
    points = [
        {"latitude": lat, "longitude": lon},
        {"latitude": lat + delta, "longitude": lon},       # North
        {"latitude": lat - delta, "longitude": lon},       # South
        {"latitude": lat, "longitude": lon + delta},       # East
        {"latitude": lat, "longitude": lon - delta},       # West
    ]

    url = "https://api.open-elevation.com/api/v1/lookup"
    resp = requests.post(url, json={"locations": points}, timeout=_TIMEOUT)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if len(results) < 5:
        return None

    center_elev = results[0]["elevation"]
    north_elev = results[1]["elevation"]
    south_elev = results[2]["elevation"]
    east_elev = results[3]["elevation"]
    west_elev = results[4]["elevation"]

    # Compute approximate slope (degrees)
    dx = delta * 111320 * math.cos(math.radians(lat))  # meters east-west
    dy = delta * 110540  # meters north-south

    dz_dx = (east_elev - west_elev) / (2 * dx)
    dz_dy = (north_elev - south_elev) / (2 * dy)
    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    # Compute elevation drop over ~1km for hydro head estimate
    elevations = [r["elevation"] for r in results]
    head_m = max(elevations) - min(elevations)

    return {
        "elevation_m": center_elev,
        "slope_degrees": slope_deg,
        "head_m": head_m,
    }


# ---------------------------------------------------------------------------
# Open-Meteo Flood API — river discharge
# https://open-meteo.com/en/docs/flood-api
# ---------------------------------------------------------------------------

def _fetch_flood_discharge(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch river discharge from Open-Meteo Flood API."""
    url = "https://flood-api.open-meteo.com/v1/flood"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "river_discharge",
        "past_days": 365,
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    discharges = daily.get("river_discharge", [])

    # Filter None values
    valid = [d for d in discharges if d is not None and d >= 0]

    if not valid:
        return None

    mean_discharge = sum(valid) / len(valid)

    return {
        "discharge_m3s": mean_discharge,
    }


# ---------------------------------------------------------------------------
# USGS Earthquake API — seismic activity for geothermal
# https://earthquake.usgs.gov/fdsnws/event/1/
# ---------------------------------------------------------------------------

def _fetch_earthquake(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch earthquake count within 100km over last 10 years from USGS."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/count"
    params = {
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": 100,
        "minmagnitude": 2.0,
        "starttime": "2016-01-01",
        "endtime": "2026-01-01",
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    count = int(resp.text.strip())

    # Convert to density: events per 10,000 km2 per year
    area_km2 = math.pi * 100**2  # ~31,416 km2
    years = 10
    density = count / (area_km2 / 10000) / years

    return {
        "earthquake_density": density,
        "earthquake_count_10yr": count,
    }
