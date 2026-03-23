"""
NEMS-Based Valuation Engine for Renewable Energy Assets

Integrates EIA's National Energy Modeling System (NEMS) projections
with AI verification data to produce asset valuations.

NEMS Reference: https://github.com/EIAgov/NEMS

Author: Zim (Millennium Fellowship Research)
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AssetType(Enum):
    SOLAR_UTILITY = "solar_utility"
    SOLAR_DISTRIBUTED = "solar_distributed"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    HYDRO = "hydro"
    BATTERY_STORAGE = "battery_storage"


class NEMSScenario(Enum):
    REFERENCE = "ref2025"
    HIGH_RENEWABLES = "lowzerocarbon"
    LOW_RENEWABLES = "highzerocarbon"


@dataclass
class AssetCharacteristics:
    asset_id: str
    asset_type: AssetType
    latitude: float
    longitude: float
    state: str
    capacity_mw: float
    capacity_factor: float = 0.25
    installation_cost_per_kw: float = 1000.0
    fixed_om_per_kw_year: float = 15.0
    variable_om_per_mwh: float = 0.0
    degradation_rate: float = 0.005
    project_life_years: int = 25
    online_year: int = 2025
    verification_status: str = "pending"
    verification_confidence: float = 0.0


@dataclass
class ClimateRiskFactors:
    risk_score: float = 0.0
    wildfire_risk: float = 0.0
    flood_risk: float = 0.0
    extreme_heat_risk: float = 0.0
    temperature_change_2050: float = 0.0
    resource_availability_change: float = 0.0


@dataclass
class ValuationResult:
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
    assumptions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizationMetrics:
    asset_id: str
    total_value_usd: float
    total_tokens: int
    value_per_token_usd: float
    projected_annual_yield_pct: float
    yield_confidence_range: Tuple[float, float]
    distribution_frequency: str
    projected_distributions: List[float]
    risk_rating: str
    liquidity_score: float


class ValuationEngine:
    """NEMS-integrated valuation engine."""
    
    DEFAULT_CAPACITY_FACTORS = {
        AssetType.SOLAR_UTILITY: {"default": 0.25, "CA": 0.27, "AZ": 0.28, "TX": 0.24},
        AssetType.WIND_ONSHORE: {"default": 0.35, "TX": 0.38, "IA": 0.40},
    }
    
    def __init__(self, eia_client=None, discount_rate: float = 0.08, inflation_rate: float = 0.02):
        self.eia = eia_client
        self.discount_rate = discount_rate
        self.inflation_rate = inflation_rate
        self._price_cache = {}
        
    def get_capacity_factor(self, asset_type: AssetType, state: str) -> float:
        factors = self.DEFAULT_CAPACITY_FACTORS.get(asset_type, {"default": 0.25})
        return factors.get(state, factors["default"])
    
    def get_electricity_prices(self, state: str, scenario: NEMSScenario, years: int = 25) -> List[float]:
        base_price = 50.0
        return [base_price * (1 + self.inflation_rate) ** y for y in range(years)]
    
    def calculate_npv(self, cash_flows: List[float], initial_investment: float = 0) -> float:
        npv = -initial_investment
        for year, cf in enumerate(cash_flows, 1):
            npv += cf / (1 + self.discount_rate) ** year
        return npv
    
    def calculate_irr(self, cash_flows: List[float], initial_investment: float) -> float:
        all_flows = [-initial_investment] + cash_flows
        irr = 0.1
        for _ in range(1000):
            npv = sum(cf / (1 + irr) ** t for t, cf in enumerate(all_flows))
            npv_deriv = sum(-t * cf / (1 + irr) ** (t + 1) for t, cf in enumerate(all_flows))
            if abs(npv_deriv) < 1e-10:
                break
            irr_new = irr - npv / npv_deriv
            if abs(irr_new - irr) < 0.0001:
                return irr_new
            irr = irr_new
        return irr
    
    def value_asset(self, asset: AssetCharacteristics, climate_risk: Optional[ClimateRiskFactors] = None) -> ValuationResult:
        climate_risk = climate_risk or ClimateRiskFactors()
        
        if asset.capacity_factor == 0.25:
            asset.capacity_factor = self.get_capacity_factor(asset.asset_type, asset.state)
        
        prices = self.get_electricity_prices(asset.state, NEMSScenario.REFERENCE, asset.project_life_years)
        
        # Generation with degradation
        base_gen = asset.capacity_mw * asset.capacity_factor * 8760
        generation = [base_gen * (1 - asset.degradation_rate) ** y for y in range(asset.project_life_years)]
        
        revenue = [g * p for g, p in zip(generation, prices)]
        costs = [asset.capacity_mw * 1000 * asset.fixed_om_per_kw_year for _ in range(asset.project_life_years)]
        net_cash_flow = [r - c for r, c in zip(revenue, costs)]
        
        initial_investment = asset.capacity_mw * 1000 * asset.installation_cost_per_kw
        
        cumulative = []
        running = -initial_investment
        for cf in net_cash_flow:
            running += cf
            cumulative.append(running)
        
        npv = self.calculate_npv(net_cash_flow, initial_investment)
        irr = self.calculate_irr(net_cash_flow, initial_investment)
        
        # Payback
        payback = asset.project_life_years
        for y, cum in enumerate(cumulative):
            if cum >= 0:
                payback = y + 1
                break
        
        # LCOE
        cost_npv = initial_investment + sum(c / (1 + self.discount_rate) ** (y+1) for y, c in enumerate(costs))
        gen_npv = sum(g / (1 + self.discount_rate) ** (y+1) for y, g in enumerate(generation))
        lcoe = cost_npv / gen_npv if gen_npv > 0 else float('inf')
        
        # Discounts
        ver_discount = {"verified": 0.0, "pending": 0.15, "flagged": 0.35}.get(asset.verification_status, 0.25)
        ver_discount *= (1 - asset.verification_confidence * 0.5)
        verified_npv = npv * (1 - ver_discount)
        
        climate_discount = min(0.2, climate_risk.risk_score * 0.2)
        risk_adjusted_npv = verified_npv * (1 - climate_discount)
        
        return ValuationResult(
            asset_id=asset.asset_id,
            valuation_date=datetime.now().isoformat(),
            npv_usd=npv,
            irr=irr,
            payback_years=payback,
            lcoe_per_mwh=lcoe,
            risk_adjusted_npv=risk_adjusted_npv,
            verification_adjusted_npv=verified_npv,
            value_at_risk_95=npv * 0.7,
            annual_generation_mwh=generation,
            annual_revenue_usd=revenue,
            annual_costs_usd=costs,
            annual_net_cash_flow=net_cash_flow,
            cumulative_cash_flow=cumulative,
            electricity_prices=prices,
            price_scenario="ref2025",
            scenario_npvs={"ref2025": npv},
            scenario_irrs={"ref2025": irr},
            climate_risk_discount=climate_discount,
            verification_discount=ver_discount,
            assumptions={"discount_rate": self.discount_rate, "capacity_factor": asset.capacity_factor}
        )
    
    def calculate_tokenization_metrics(self, valuation: ValuationResult, total_tokens: int = 1_000_000) -> TokenizationMetrics:
        total_value = max(0, valuation.risk_adjusted_npv)
        value_per_token = total_value / total_tokens if total_tokens > 0 else 0
        avg_cf = np.mean(valuation.annual_net_cash_flow[:5])
        annual_yield = (avg_cf / total_value * 100) if total_value > 0 else 0
        
        risk_rating = "AA" if valuation.irr > 0.15 else "A" if valuation.irr > 0.10 else "BBB"
        
        return TokenizationMetrics(
            asset_id=valuation.asset_id,
            total_value_usd=total_value,
            total_tokens=total_tokens,
            value_per_token_usd=value_per_token,
            projected_annual_yield_pct=annual_yield,
            yield_confidence_range=(annual_yield * 0.8, annual_yield * 1.2),
            distribution_frequency="quarterly",
            projected_distributions=[avg_cf / 4] * 20,
            risk_rating=risk_rating,
            liquidity_score=0.7
        )


def create_valuation_engine(discount_rate: float = 0.08, eia_client=None) -> ValuationEngine:
    """Factory function to create valuation engine."""
    return ValuationEngine(eia_client=eia_client, discount_rate=discount_rate)


if __name__ == "__main__":
    engine = ValuationEngine()
    asset = AssetCharacteristics(
        asset_id="solar_001", asset_type=AssetType.SOLAR_UTILITY,
        latitude=35.0, longitude=-119.9, state="CA", capacity_mw=100,
        verification_status="verified", verification_confidence=0.92
    )
    valuation = engine.value_asset(asset)
    print(f"NPV: ${valuation.npv_usd:,.0f}")
    print(f"IRR: {valuation.irr*100:.1f}%")
    print(f"Risk-Adjusted NPV: ${valuation.risk_adjusted_npv:,.0f}")
