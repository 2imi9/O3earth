"""EIA data query routes."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.mcp.tools import EIAQueryInput

router = APIRouter()


def _get_handlers():
    from api.main import handlers
    if handlers is None:
        raise HTTPException(503, "Service not ready")
    return handlers


@router.get("/eia/generators")
async def get_generators(
    state: Optional[str] = Query(None, description="US state code"),
    energy_source: Optional[str] = Query(None, description="SUN, WND, WAT, etc."),
    min_capacity_mw: float = Query(1.0, ge=0),
):
    """Query EIA generator inventory."""
    handlers = _get_handlers()
    result = handlers.query_eia(
        EIAQueryInput(
            query_type="generators",
            state=state,
            energy_source=energy_source,
            min_capacity_mw=min_capacity_mw,
        )
    )
    if "error" in result:
        raise HTTPException(400, result)
    return result


@router.get("/eia/generation/{state}")
async def get_generation(
    state: str,
    energy_source: Optional[str] = Query(None),
):
    """Query EIA generation data for a state."""
    handlers = _get_handlers()
    result = handlers.query_eia(
        EIAQueryInput(
            query_type="generation",
            state=state,
            energy_source=energy_source,
        )
    )
    if "error" in result:
        raise HTTPException(400, result)
    return result


@router.get("/eia/prices")
async def get_prices(
    scenario: str = Query("ref2025", description="AEO scenario"),
):
    """Query EIA electricity price forecasts."""
    handlers = _get_handlers()
    result = handlers.query_eia(
        EIAQueryInput(query_type="prices", scenario=scenario)
    )
    if "error" in result:
        raise HTTPException(400, result)
    return result


@router.get("/eia/capacity/{energy_source}")
async def get_capacity(
    energy_source: str,
    scenario: str = Query("ref2025"),
):
    """Query EIA renewable capacity forecasts."""
    handlers = _get_handlers()
    result = handlers.query_eia(
        EIAQueryInput(
            query_type="capacity",
            energy_source=energy_source,
            scenario=scenario,
        )
    )
    if "error" in result:
        raise HTTPException(400, result)
    return result


@router.get("/eia/summary/{state}")
async def get_summary(state: str):
    """Get renewable energy summary for a state."""
    handlers = _get_handlers()
    result = handlers.query_eia(
        EIAQueryInput(query_type="summary", state=state)
    )
    if "error" in result:
        raise HTTPException(400, result)
    return result
