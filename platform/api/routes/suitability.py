"""Suitability route — site suitability scoring using OlmoEarth embeddings."""

import sys
from pathlib import Path

# Ensure project root is available
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
    energy_type: str = "solar"  # solar, wind, hydro, geothermal


class SuitabilityResponse(BaseModel):
    suitability_score: float
    energy_type: str
    latitude: float
    longitude: float
    confidence: str  # high, moderate, low
    factors: dict


@router.post("/suitability", response_model=SuitabilityResponse)
async def score_suitability(req: SuitabilityRequest):
    """Score a location for renewable energy suitability."""
    try:
        # Use the factor engine for a quick score
        from src.scoring.engine import SuitabilityEngine

        engine = SuitabilityEngine(req.energy_type)
        result = engine.score(req.latitude, req.longitude)

        score = result.overall_score
        if score > 0.7:
            confidence = "high"
        elif score > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return SuitabilityResponse(
            suitability_score=score,
            energy_type=req.energy_type,
            latitude=req.latitude,
            longitude=req.longitude,
            confidence=confidence,
            factors=result.factor_scores,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
