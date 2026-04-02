"""Climate risk assessment route — using real NASA POWER data."""

import sys
import math
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

router = APIRouter()


class ClimateRiskRequest(BaseModel):
    latitude: float
    longitude: float
    elevation: float = 0.0
    asset_type: str = "solar"
    scenario: str = "SSP245"
    target_year: int = 2050


class ClimateRiskResponse(BaseModel):
    latitude: float
    longitude: float
    risk_score: float
    solar_ghi_kwh_m2_year: dict
    wind_speed_m_s: dict
    extreme_event_probs: dict
    temperature_change_c: float
    precipitation_change_pct: float
    confidence: float
    uncertainty_range: list


# SSP scenario temperature projections (IPCC AR6 global mean estimates)
SSP_TEMP_CHANGE = {
    "SSP126": {2030: 0.6, 2050: 1.0, 2070: 1.2, 2100: 1.4},
    "SSP245": {2030: 0.7, 2050: 1.4, 2070: 2.0, 2100: 2.7},
    "SSP370": {2030: 0.7, 2050: 1.6, 2070: 2.5, 2100: 3.6},
    "SSP585": {2030: 0.8, 2050: 1.9, 2070: 3.1, 2100: 4.4},
}

SSP_PRECIP_CHANGE = {
    "SSP126": {2030: 1.0, 2050: 2.0, 2070: 2.5, 2100: 3.0},
    "SSP245": {2030: 1.5, 2050: 3.0, 2070: 4.5, 2100: 5.5},
    "SSP370": {2030: 1.5, 2050: 3.5, 2070: 5.5, 2100: 7.0},
    "SSP585": {2030: 2.0, 2050: 4.5, 2070: 7.0, 2100: 9.0},
}


def _interpolate_projection(table: dict, year: int) -> float:
    """Interpolate between projection decades."""
    years = sorted(table.keys())
    if year <= years[0]:
        return table[years[0]]
    if year >= years[-1]:
        return table[years[-1]]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            frac = (year - years[i]) / (years[i + 1] - years[i])
            return table[years[i]] + frac * (table[years[i + 1]] - table[years[i]])
    return table[years[-1]]


@router.post("/climate-risk", response_model=ClimateRiskResponse)
async def assess_climate_risk(req: ClimateRiskRequest):
    """Assess climate risk using real NASA POWER data + IPCC projections."""
    try:
        from src.data_clients.realtime import _fetch_nasa_power

        # 1. Get real current climate data
        nasa_data = None
        try:
            nasa_data = _fetch_nasa_power(req.latitude, req.longitude)
        except Exception:
            pass

        # Current values (from NASA POWER or estimates)
        current_ghi = (nasa_data or {}).get("ghi_kwh_m2_day", 4.5)
        current_wind = (nasa_data or {}).get("wind_speed_ms", 5.0)
        current_temp = (nasa_data or {}).get("avg_temp_c", 15.0)
        current_cloud = (nasa_data or {}).get("cloud_fraction", 0.5)
        current_precip = (nasa_data or {}).get("precipitation_mm_day", 2.0)

        # Annual GHI
        ghi_annual = current_ghi * 365

        # 2. Apply SSP projections
        scenario = req.scenario.upper()
        if scenario not in SSP_TEMP_CHANGE:
            scenario = "SSP245"

        temp_change = _interpolate_projection(SSP_TEMP_CHANGE[scenario], req.target_year)
        precip_change = _interpolate_projection(SSP_PRECIP_CHANGE[scenario], req.target_year)

        # GHI adjustment: slight decrease with warming (more moisture/clouds)
        ghi_factor = 1.0 - (temp_change * 0.005)  # ~0.5% decrease per degree
        projected_ghi = ghi_annual * ghi_factor

        # Wind adjustment: minor changes with warming
        wind_factor = 1.0 + (temp_change * 0.01)  # slight increase
        projected_wind = current_wind * wind_factor

        # 3. Extreme event probabilities based on location + scenario
        abs_lat = abs(req.latitude)
        # Tropical cyclone risk
        cyclone_risk = 0.0
        if abs_lat < 30 and abs_lat > 5:
            cyclone_risk = 0.15 * (1 + temp_change * 0.1)

        # Extreme heat risk
        heat_risk = 0.0
        if current_temp > 20:
            heat_risk = min(0.8, (current_temp + temp_change - 25) * 0.05)
        heat_risk = max(0.0, heat_risk)

        # Flood risk
        flood_risk = min(0.5, current_precip * 0.03 * (1 + precip_change / 100))

        # Drought risk (inverse of precipitation)
        drought_risk = max(0.0, 0.5 - current_precip * 0.05) * (1 + temp_change * 0.05)

        extreme_probs = {
            "tropical_cyclone": round(cyclone_risk, 3),
            "extreme_heat": round(heat_risk, 3),
            "flooding": round(flood_risk, 3),
            "drought": round(drought_risk, 3),
        }

        # 4. Overall risk score (0 = low risk, 1 = high risk)
        risk_score = (
            0.3 * heat_risk
            + 0.25 * flood_risk
            + 0.25 * cyclone_risk
            + 0.2 * drought_risk
        )
        # Amplify by scenario severity
        scenario_weight = {"SSP126": 0.8, "SSP245": 1.0, "SSP370": 1.2, "SSP585": 1.5}
        risk_score = min(1.0, risk_score * scenario_weight.get(scenario, 1.0))

        # 5. Confidence based on data availability
        confidence = 0.85 if nasa_data else 0.5

        # Uncertainty range
        uncertainty = 0.1 + temp_change * 0.02
        uncertainty_range = [max(0, risk_score - uncertainty), min(1, risk_score + uncertainty)]

        return ClimateRiskResponse(
            latitude=req.latitude,
            longitude=req.longitude,
            risk_score=round(risk_score, 3),
            solar_ghi_kwh_m2_year={
                "p10": round(projected_ghi * 0.9, 0),
                "p50": round(projected_ghi, 0),
                "p90": round(projected_ghi * 1.1, 0),
            },
            wind_speed_m_s={
                "p10": round(projected_wind * 0.85, 1),
                "p50": round(projected_wind, 1),
                "p90": round(projected_wind * 1.15, 1),
            },
            extreme_event_probs=extreme_probs,
            temperature_change_c=round(temp_change, 1),
            precipitation_change_pct=round(precip_change, 1),
            confidence=round(confidence, 2),
            uncertainty_range=[round(x, 3) for x in uncertainty_range],
        )
    except Exception as e:
        raise HTTPException(500, str(e))
