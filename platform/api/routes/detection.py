"""Detection route — satellite-based renewable energy detection."""

from fastapi import APIRouter, HTTPException
from dataclasses import asdict

from api.schemas import DetectionRequest, DetectionResponse
from src.mcp.tools import DetectionInput

router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_renewable(req: DetectionRequest):
    """Detect renewable energy installations from satellite imagery."""
    from api.main import handlers

    if handlers is None:
        raise HTTPException(503, "Service not ready")

    try:
        result = handlers.detect_renewable(
            DetectionInput(
                latitude=req.latitude,
                longitude=req.longitude,
                date_range=req.date_range,
                max_cloud_cover=req.max_cloud_cover,
            )
        )
        return DetectionResponse(**asdict(result))
    except Exception as e:
        raise HTTPException(500, str(e))
