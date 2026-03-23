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
    factor_score: float
    ml_score: Optional[float] = None
    combined_score: float
    energy_type: str
    latitude: float
    longitude: float
    confidence: str
    factors: dict
    ml_available: bool = False


@router.post("/suitability", response_model=SuitabilityResponse)
async def score_suitability(req: SuitabilityRequest):
    """Score a location for renewable energy suitability.

    Returns both factor engine score and ML model score (if embeddings available).
    """
    try:
        # 1. Factor engine score (always available, instant)
        from src.scoring.engine import SuitabilityEngine

        engine = SuitabilityEngine(req.energy_type)
        result = engine.score(req.latitude, req.longitude)
        factor_score = result.overall_score

        # 2. ML model score (needs OlmoEarth embeddings)
        ml_score = None
        ml_available = False
        try:
            import xgboost as xgb
            import numpy as np
            import pickle

            model_path = Path(_project_root) / f"results/suitability/xgb_{req.energy_type}.json"
            scaler_path = Path(_project_root) / f"results/suitability/scaler_{req.energy_type}.pkl"

            if model_path.exists() and scaler_path.exists():
                # Check if we have a pre-computed embedding for nearby location
                meta_path = Path(_project_root) / "data/embeddings_shuffled/embeddings_meta.csv"
                emb_path = Path(_project_root) / "data/embeddings_shuffled/embeddings.npy"

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

        # 3. Combined score
        if ml_score is not None:
            combined_score = 0.6 * ml_score + 0.4 * factor_score
        else:
            combined_score = factor_score

        if combined_score > 0.7:
            confidence = "high"
        elif combined_score > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"

        return SuitabilityResponse(
            factor_score=factor_score,
            ml_score=ml_score,
            combined_score=combined_score,
            energy_type=req.energy_type,
            latitude=req.latitude,
            longitude=req.longitude,
            confidence=confidence,
            factors=result.factor_scores,
            ml_available=ml_available,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
