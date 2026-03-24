"""Suitability route — site suitability scoring with real-time data."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class SuitabilityRequest(BaseModel):
    latitude: float
    longitude: float
    energy_type: str = "solar"


class SuitabilityResponse(BaseModel):
    factor_score: float
    combined_score: float
    energy_type: str
    latitude: float
    longitude: float
    confidence: str
    factors: dict
    data_sources: dict = {}


@router.post("/suitability", response_model=SuitabilityResponse)
async def score_suitability(req: SuitabilityRequest):
    """Score a location for renewable energy suitability using real-time data."""
    try:
        from src.scoring.engine import SuitabilityEngine
        from src.data_clients.realtime import fetch_all

        # 1. Fetch real data from NASA POWER, USGS, Open-Meteo, etc.
        real_data = fetch_all(req.latitude, req.longitude)

        # 2. Run factor engine with real data as kwargs
        engine = SuitabilityEngine(req.energy_type)
        result = engine.score(req.latitude, req.longitude, **real_data)
        factor_score = result.overall_score

        # Track which data sources returned values
        data_sources = {}
        source_map = {
            "ghi_kwh_m2_day": "NASA POWER",
            "wind_speed_ms": "NASA POWER",
            "avg_temp_c": "NASA POWER",
            "cloud_fraction": "NASA POWER",
            "precipitation_mm_day": "NASA POWER",
            "elevation_m": "Open-Elevation",
            "slope_degrees": "Open-Elevation",
            "head_m": "Open-Elevation",
            "discharge_m3s": "Open-Meteo Flood",
            "earthquake_density": "USGS Earthquake",
        }
        for key, source in source_map.items():
            if key in real_data:
                data_sources[key] = {"source": source, "value": round(real_data[key], 3)}

        combined_score = factor_score

        if combined_score > 0.7:
            confidence = "high"
        elif combined_score > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return SuitabilityResponse(
            factor_score=factor_score,
            combined_score=combined_score,
            energy_type=req.energy_type,
            latitude=req.latitude,
            longitude=req.longitude,
            confidence=confidence,
            factors=result.factor_scores,
            data_sources=data_sources,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
