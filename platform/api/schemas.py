"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Tuple


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class DetectionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude in degrees")
    longitude: float = Field(..., description="Longitude in degrees")
    date_range: str = Field("2024-01-01/2024-12-31", description="YYYY-MM-DD/YYYY-MM-DD")
    max_cloud_cover: float = Field(10.0, ge=0, le=100)


class ClimateRiskRequest(BaseModel):
    latitude: float
    longitude: float
    elevation: float = 0.0
    asset_type: str = Field("solar", pattern="^(solar|wind)$")
    scenario: str = Field("SSP245", pattern="^SSP(126|245|370|585)$")
    target_year: int = Field(2050, ge=2025, le=2100)


class ValuationRequest(BaseModel):
    asset_id: str = "asset_001"
    asset_type: str = Field("solar_utility", description="One of: solar_utility, solar_distributed, wind_onshore, wind_offshore, hydro, battery_storage")
    latitude: float = 35.0
    longitude: float = -119.9
    state: str = Field("CA", min_length=2, max_length=2)
    capacity_mw: float = Field(100.0, gt=0)
    capacity_factor: float = Field(0.25, ge=0.0, le=1.0)
    installation_cost_per_kw: float = 1000.0
    fixed_om_per_kw_year: float = 15.0
    degradation_rate: float = 0.005
    project_life_years: int = 25
    verification_status: str = Field("pending", pattern="^(verified|pending|flagged)$")
    verification_confidence: float = Field(0.0, ge=0.0, le=1.0)
    discount_rate: float = Field(0.08, ge=0.01, le=0.25)


class EIAQueryRequest(BaseModel):
    query_type: str = Field(..., pattern="^(generators|generation|prices|capacity|summary)$")
    state: Optional[str] = None
    energy_source: Optional[str] = None
    min_capacity_mw: float = 1.0
    scenario: str = "ref2025"


class TokenizationRequest(BaseModel):
    valuation_data: Dict[str, Any]
    total_tokens: int = Field(1_000_000, gt=0)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class DetectionResponse(BaseModel):
    detected: bool
    detection_confidence: float
    classification: str
    classification_confidence: float
    estimated_capacity_mw: float
    image_date: str
    image_source: str


class ClimateRiskResponse(BaseModel):
    latitude: float
    longitude: float
    risk_score: float
    solar_ghi_kwh_m2_year: Dict[str, float]
    wind_speed_m_s: Dict[str, float]
    extreme_event_probs: Dict[str, float]
    temperature_change_c: float
    precipitation_change_pct: float
    confidence: float
    uncertainty_range: List[float]


class ValuationResponse(BaseModel):
    asset_id: str
    valuation_date: str
    npv_usd: float
    irr: float
    payback_years: float
    lcoe_per_mwh: float
    risk_adjusted_npv: float
    verification_adjusted_npv: float
    value_at_risk_95: float
    annual_generation_mwh: List[float]
    annual_revenue_usd: List[float]
    annual_costs_usd: List[float]
    annual_net_cash_flow: List[float]
    cumulative_cash_flow: List[float]
    electricity_prices: List[float]
    price_scenario: str
    scenario_npvs: Dict[str, float]
    scenario_irrs: Dict[str, float]
    climate_risk_discount: float
    verification_discount: float
    assumptions: Dict[str, Any]


class TokenizationResponse(BaseModel):
    asset_id: str
    total_value_usd: float
    total_tokens: int
    value_per_token_usd: float
    projected_annual_yield_pct: float
    yield_confidence_range: List[float]
    distribution_frequency: str
    projected_distributions: List[float]
    risk_rating: str
    liquidity_score: float


class HealthResponse(BaseModel):
    status: str
    modules: Dict[str, bool]
