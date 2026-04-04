"""Suitability route — dual scoring: Factor Engine + OlmoEarth ML."""

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
    ml_score: Optional[float] = None
    energy_type: str
    latitude: float
    longitude: float
    confidence: str
    factors: dict
    data_sources: dict = {}
    ml_available: bool = False


@router.post("/suitability", response_model=SuitabilityResponse)
async def score_suitability(req: SuitabilityRequest):
    """Score a location using both Factor Engine and OlmoEarth ML model."""
    try:
        from src.scoring.engine import SuitabilityEngine
        from src.data_clients.realtime import fetch_all

        # 1. Factor engine score (real-time API data)
        real_data = fetch_all(req.latitude, req.longitude)
        engine = SuitabilityEngine(req.energy_type)
        result = engine.score(req.latitude, req.longitude, **real_data)
        factor_score = result.overall_score

        # Track data sources
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

        # 2. ML model score (OlmoEarth embeddings + XGBoost)
        ml_score = None
        ml_available = False
        try:
            import xgboost as xgb
            import numpy as np
            import pickle

            model_path = Path(_project_root) / f"results/suitability/xgb_{req.energy_type}.json"
            scaler_path = Path(_project_root) / f"results/suitability/scaler_{req.energy_type}.pkl"

            if model_path.exists() and scaler_path.exists():
                meta_path = Path(_project_root) / "data/embeddings_v3/embeddings_meta.csv"
                emb_path = Path(_project_root) / "data/embeddings_v3/embeddings.npy"

                if meta_path.exists() and emb_path.exists():
                    import pandas as pd
                    meta = pd.read_csv(str(meta_path))
                    emb = np.load(str(emb_path))

                    # Find nearest embedding within 0.5 degrees
                    dist = np.sqrt((meta["lat"] - req.latitude)**2 + (meta["lon"] - req.longitude)**2)
                    nearest_idx = dist.idxmin()
                    nearest_dist = dist[nearest_idx]

                    if nearest_dist < 0.5:
                        model = xgb.XGBClassifier()
                        model.load_model(str(model_path))
                        with open(str(scaler_path), "rb") as f:
                            scaler = pickle.load(f)

                        embedding = emb[nearest_idx].reshape(1, -1)
                        embedding_s = scaler.transform(embedding)
                        ml_score = float(model.predict_proba(embedding_s)[0, 1])
                        ml_available = True
        except Exception:
            pass

        # Confidence based on factor score
        if factor_score > 0.7:
            confidence = "high"
        elif factor_score > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return SuitabilityResponse(
            factor_score=factor_score,
            ml_score=ml_score,
            energy_type=req.energy_type,
            latitude=req.latitude,
            longitude=req.longitude,
            confidence=confidence,
            factors=result.factor_scores,
            data_sources=data_sources,
            ml_available=ml_available,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
