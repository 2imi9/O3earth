"""Climate risk assessment route."""

from fastapi import APIRouter, HTTPException
from dataclasses import asdict

from api.schemas import ClimateRiskRequest, ClimateRiskResponse
from src.mcp.tools import ClimateRiskInput

router = APIRouter()


@router.post("/climate-risk", response_model=ClimateRiskResponse)
async def assess_climate_risk(req: ClimateRiskRequest):
    """Assess climate risk and resource availability for a location."""
    from api.main import handlers

    if handlers is None:
        raise HTTPException(503, "Service not ready")

    try:
        tool_result = handlers.assess_climate_risk(
            ClimateRiskInput(
                latitude=req.latitude,
                longitude=req.longitude,
                elevation=req.elevation,
                asset_type=req.asset_type,
                scenario=req.scenario,
                target_year=req.target_year,
            )
        )

        # The ToolHandlers returns a ClimateRiskOutput (flat).
        # We need the richer ClimateRiskOutput from the model for GHI P10/P90.
        # Re-run through the model directly for the full output.
        from src.models import create_climate_model, ClimateScenario

        model = create_climate_model()
        scenario_map = {
            "SSP126": ClimateScenario.SSP126,
            "SSP245": ClimateScenario.SSP245,
            "SSP370": ClimateScenario.SSP370,
            "SSP585": ClimateScenario.SSP585,
        }
        scenario = scenario_map.get(req.scenario.upper(), ClimateScenario.SSP245)

        risk = model.assess_risk(
            latitude=req.latitude,
            longitude=req.longitude,
            elevation=req.elevation,
            asset_type=req.asset_type,
            scenario=scenario,
            target_year=req.target_year,
        )

        return ClimateRiskResponse(
            latitude=risk.latitude,
            longitude=risk.longitude,
            risk_score=risk.risk_score,
            solar_ghi_kwh_m2_year=risk.solar_ghi_kwh_m2_year,
            wind_speed_m_s=risk.wind_speed_m_s,
            extreme_event_probs=risk.extreme_event_probs,
            temperature_change_c=risk.temperature_change_c,
            precipitation_change_pct=risk.precipitation_change_pct,
            confidence=risk.confidence,
            uncertainty_range=list(risk.uncertainty_range),
        )
    except Exception as e:
        raise HTTPException(500, str(e))
