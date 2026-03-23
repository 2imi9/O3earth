"""Asset valuation route — returns full 25-year projections for charting."""

from fastapi import APIRouter, HTTPException
from dataclasses import asdict

from api.schemas import ValuationRequest, ValuationResponse, TokenizationRequest, TokenizationResponse
from src.valuation import ValuationEngine, AssetCharacteristics, AssetType

router = APIRouter()

# Cache engines by discount rate
_engines: dict[float, ValuationEngine] = {}


def _get_engine(discount_rate: float) -> ValuationEngine:
    if discount_rate not in _engines:
        _engines[discount_rate] = ValuationEngine(discount_rate=discount_rate)
    return _engines[discount_rate]


@router.post("/value-asset", response_model=ValuationResponse)
async def value_asset(req: ValuationRequest):
    """Calculate full NPV/IRR/LCOE valuation with 25-year cash flow projections."""
    type_map = {
        "solar_utility": AssetType.SOLAR_UTILITY,
        "solar_distributed": AssetType.SOLAR_DISTRIBUTED,
        "wind_onshore": AssetType.WIND_ONSHORE,
        "wind_offshore": AssetType.WIND_OFFSHORE,
        "hydro": AssetType.HYDRO,
        "battery_storage": AssetType.BATTERY_STORAGE,
    }

    asset_type = type_map.get(req.asset_type.lower())
    if asset_type is None:
        raise HTTPException(400, f"Unknown asset type: {req.asset_type}")

    try:
        engine = _get_engine(req.discount_rate)
        asset = AssetCharacteristics(
            asset_id=req.asset_id,
            asset_type=asset_type,
            latitude=req.latitude,
            longitude=req.longitude,
            state=req.state,
            capacity_mw=req.capacity_mw,
            capacity_factor=req.capacity_factor,
            installation_cost_per_kw=req.installation_cost_per_kw,
            fixed_om_per_kw_year=req.fixed_om_per_kw_year,
            degradation_rate=req.degradation_rate,
            project_life_years=req.project_life_years,
            verification_status=req.verification_status,
            verification_confidence=req.verification_confidence,
        )
        result = engine.value_asset(asset)
        return ValuationResponse(**asdict(result))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/tokenize", response_model=TokenizationResponse)
async def tokenize_asset(req: TokenizationRequest):
    """Calculate tokenization metrics from a valuation result."""
    from src.valuation import ValuationResult

    try:
        engine = _get_engine(0.08)

        # Reconstruct ValuationResult from dict
        vr = ValuationResult(**req.valuation_data)
        metrics = engine.calculate_tokenization_metrics(vr, req.total_tokens)

        return TokenizationResponse(
            asset_id=metrics.asset_id,
            total_value_usd=metrics.total_value_usd,
            total_tokens=metrics.total_tokens,
            value_per_token_usd=metrics.value_per_token_usd,
            projected_annual_yield_pct=metrics.projected_annual_yield_pct,
            yield_confidence_range=list(metrics.yield_confidence_range),
            distribution_frequency=metrics.distribution_frequency,
            projected_distributions=metrics.projected_distributions,
            risk_rating=metrics.risk_rating,
            liquidity_score=metrics.liquidity_score,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
