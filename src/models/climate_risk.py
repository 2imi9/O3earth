"""
Climate Risk Model for Renewable Energy Infrastructure

Integrates climate projections with renewable energy asset assessment:
- Solar irradiance forecasting
- Wind resource assessment
- Extreme event risk (wildfire, flood, heat)
- Climate change impacts on capacity factors

Based on concepts from:
- ACE2 Climate Emulator (Allen Institute)
- NEMS projections (EIA)
- Ganguly et al. climate adaptation informatics

Author: Zim (Millennium Fellowship Research)
Project: AI Earth Observation for Democratic Sustainable Energy Investment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ClimateScenario(Enum):
    """IPCC SSP scenarios for climate projections."""
    SSP126 = "ssp126"  # Sustainability
    SSP245 = "ssp245"  # Middle of the road
    SSP370 = "ssp370"  # Regional rivalry
    SSP585 = "ssp585"  # Fossil-fueled development


class ExtremeEventType(Enum):
    """Types of extreme events affecting energy infrastructure."""
    WILDFIRE = "wildfire"
    FLOOD = "flood"
    DROUGHT = "drought"
    EXTREME_HEAT = "extreme_heat"
    EXTREME_COLD = "extreme_cold"
    HURRICANE = "hurricane"
    HAIL = "hail"


@dataclass
class ClimateConfig:
    """Configuration for climate risk model."""
    
    # Spatial resolution
    grid_resolution_km: float = 25.0  # ~0.25 degrees
    
    # Temporal
    forecast_years: int = 30
    base_year: int = 2025
    
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    
    # Input variables (ERA5-style)
    surface_variables: List[str] = field(default_factory=lambda: [
        "t2m",      # 2m temperature
        "tp",       # Total precipitation
        "ssrd",     # Surface solar radiation downward
        "u10",      # 10m U wind
        "v10",      # 10m V wind
        "sp",       # Surface pressure
        "tcwv",     # Total column water vapor
    ])
    
    # Pressure levels for atmospheric variables
    pressure_levels: List[int] = field(default_factory=lambda: [
        1000, 925, 850, 700, 500, 300, 200, 100, 50
    ])
    
    atmospheric_variables: List[str] = field(default_factory=lambda: [
        "t",        # Temperature
        "u",        # U wind
        "v",        # V wind
        "q",        # Specific humidity
        "z",        # Geopotential
    ])


@dataclass
class ClimateRiskOutput:
    """Output of climate risk assessment."""
    
    # Location
    latitude: float
    longitude: float
    
    # Overall risk score (0-1, higher = more risk)
    risk_score: float
    
    # Resource assessment
    solar_ghi_kwh_m2_year: Dict[str, float]  # P10, P50, P90
    wind_speed_m_s: Dict[str, float]         # P10, P50, P90
    capacity_factor_projection: List[float]   # By year
    
    # Extreme event probabilities (annual)
    extreme_event_probs: Dict[str, float]
    
    # Climate change impacts
    temperature_change_c: float              # Projected change by 2050
    precipitation_change_pct: float          # Projected change by 2050
    
    # Confidence
    confidence: float
    uncertainty_range: Tuple[float, float]


class SolarResourceEncoder(nn.Module):
    """Encode solar resource variables for GHI prediction."""
    
    def __init__(self, config: ClimateConfig):
        super().__init__()
        
        # Solar-specific inputs: lat, lon, elevation, cloud cover, aerosol
        self.encoder = nn.Sequential(
            nn.Linear(10, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Monthly pattern embedding (seasonal)
        self.month_embed = nn.Embedding(12, config.hidden_dim // 4)
        
    def forward(
        self, 
        location: torch.Tensor,      # (B, 3) lat, lon, elevation
        climate_vars: torch.Tensor,  # (B, 7) surface variables
        month: torch.Tensor          # (B,) month index
    ) -> torch.Tensor:
        """
        Returns:
            (B, D) solar resource encoding
        """
        # Combine inputs
        x = torch.cat([location, climate_vars], dim=-1)
        x = self.encoder(x)
        
        # Add seasonal pattern
        month_emb = self.month_embed(month)
        x[:, :month_emb.shape[-1]] = x[:, :month_emb.shape[-1]] + month_emb
        
        return x


class WindResourceEncoder(nn.Module):
    """Encode wind resource variables."""
    
    def __init__(self, config: ClimateConfig):
        super().__init__()
        
        # Wind-specific: lat, lon, elevation, roughness, u10, v10, pressure
        self.encoder = nn.Sequential(
            nn.Linear(12, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Hub height extrapolation layer
        self.height_extrap = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(
        self,
        location: torch.Tensor,      # (B, 3) lat, lon, elevation
        climate_vars: torch.Tensor,  # (B, 7) surface variables
        roughness: torch.Tensor,     # (B, 1) surface roughness
        hub_height: torch.Tensor     # (B, 1) turbine hub height in meters
    ) -> torch.Tensor:
        """
        Returns:
            (B, D) wind resource encoding at hub height
        """
        # Combine inputs
        x = torch.cat([location, climate_vars, roughness, hub_height], dim=-1)
        x = self.encoder(x)
        
        # Adjust for hub height
        x = torch.cat([x, hub_height / 100.0], dim=-1)  # Normalize height
        x = self.height_extrap(x)
        
        return x


class ExtremeEventPredictor(nn.Module):
    """Predict probability of extreme events."""
    
    def __init__(self, config: ClimateConfig):
        super().__init__()
        
        self.num_event_types = len(ExtremeEventType)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Event-specific heads
        self.event_heads = nn.ModuleDict({
            event.value: nn.Linear(config.hidden_dim // 2, 1)
            for event in ExtremeEventType
        })
        
    def forward(self, climate_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            climate_encoding: (B, D) climate state encoding
        Returns:
            Dict of event type -> probability
        """
        x = self.encoder(climate_encoding)
        
        probs = {}
        for event in ExtremeEventType:
            logit = self.event_heads[event.value](x)
            probs[event.value] = torch.sigmoid(logit).squeeze(-1)
            
        return probs


class ClimateProjector(nn.Module):
    """Project climate variables into the future under different scenarios."""
    
    def __init__(self, config: ClimateConfig):
        super().__init__()
        self.config = config
        
        # Scenario embeddings
        self.scenario_embed = nn.Embedding(len(ClimateScenario), config.hidden_dim)
        
        # Year embedding (normalized)
        self.year_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.hidden_dim)
        )
        
        # Projection network
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Output heads for delta predictions
        self.temp_delta = nn.Linear(config.hidden_dim, 1)
        self.precip_delta = nn.Linear(config.hidden_dim, 1)
        self.solar_delta = nn.Linear(config.hidden_dim, 1)
        self.wind_delta = nn.Linear(config.hidden_dim, 1)
        
    def forward(
        self,
        baseline_encoding: torch.Tensor,  # (B, D) current climate state
        scenario: torch.Tensor,           # (B,) scenario index
        target_year: torch.Tensor         # (B,) target year
    ) -> Dict[str, torch.Tensor]:
        """
        Project climate deltas for a future year under given scenario.
        
        Returns:
            Dict with temperature, precipitation, solar, wind deltas
        """
        # Encode scenario
        scenario_emb = self.scenario_embed(scenario)
        
        # Encode year (normalized to 0-1 range over projection period)
        year_norm = (target_year - self.config.base_year).float() / self.config.forecast_years
        year_emb = self.year_encoder(year_norm.unsqueeze(-1))
        
        # Combine
        x = torch.cat([baseline_encoding + scenario_emb, year_emb], dim=-1)
        x = self.projector(x)
        
        return {
            "temperature_delta_c": self.temp_delta(x).squeeze(-1),
            "precipitation_delta_pct": self.precip_delta(x).squeeze(-1) * 100,
            "solar_delta_pct": self.solar_delta(x).squeeze(-1) * 100,
            "wind_delta_pct": self.wind_delta(x).squeeze(-1) * 100
        }


class ClimateRiskModel(nn.Module):
    """
    Complete climate risk model for renewable energy infrastructure.
    
    Assesses:
    1. Solar/wind resource availability
    2. Extreme event exposure
    3. Climate change impacts on capacity factors
    4. Overall risk scoring
    
    Based on concepts from ACE2 climate emulator and NEMS projections.
    
    Usage:
        config = ClimateConfig()
        model = ClimateRiskModel(config)
        
        risk = model.assess_risk(
            latitude=35.0,
            longitude=-119.9,
            elevation=500,
            asset_type="solar",
            scenario=ClimateScenario.SSP245
        )
    """
    
    def __init__(self, config: ClimateConfig):
        super().__init__()
        self.config = config
        
        # Climate state encoder
        num_surface = len(config.surface_variables)
        num_atmos = len(config.atmospheric_variables) * len(config.pressure_levels)
        
        self.climate_encoder = nn.Sequential(
            nn.Linear(num_surface + num_atmos + 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # Resource encoders
        self.solar_encoder = SolarResourceEncoder(config)
        self.wind_encoder = WindResourceEncoder(config)
        
        # Extreme event predictor
        self.extreme_predictor = ExtremeEventPredictor(config)
        
        # Climate projector
        self.climate_projector = ClimateProjector(config)
        
        # Risk aggregation
        self.risk_aggregator = nn.Sequential(
            nn.Linear(config.hidden_dim + len(ExtremeEventType), config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Resource prediction heads
        self.ghi_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),  # P10, P50, P90
            nn.Softplus()
        )
        
        self.wind_speed_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),  # P10, P50, P90
            nn.Softplus()
        )
        
    def encode_climate_state(
        self,
        location: torch.Tensor,
        surface_vars: torch.Tensor,
        atmospheric_vars: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode current climate state."""
        if atmospheric_vars is not None:
            x = torch.cat([location, surface_vars, atmospheric_vars.flatten(-2)], dim=-1)
        else:
            x = torch.cat([location, surface_vars], dim=-1)
            # Pad if no atmospheric data
            padding = torch.zeros(
                x.shape[0], 
                len(self.config.atmospheric_variables) * len(self.config.pressure_levels),
                device=x.device
            )
            x = torch.cat([x, padding], dim=-1)
            
        return self.climate_encoder(x)
    
    def forward(
        self,
        location: torch.Tensor,           # (B, 3) lat, lon, elevation
        surface_vars: torch.Tensor,       # (B, 7) surface climate variables
        scenario: torch.Tensor,           # (B,) SSP scenario index
        target_year: torch.Tensor,        # (B,) target year
        atmospheric_vars: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        hub_height: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for climate risk assessment.
        
        Returns:
            Dict with risk score, resource predictions, extreme event probs
        """
        B = location.shape[0]
        device = location.device
        
        # Default values
        if roughness is None:
            roughness = torch.ones(B, 1, device=device) * 0.1
        if hub_height is None:
            hub_height = torch.ones(B, 1, device=device) * 100  # 100m default
        
        # Encode climate state
        climate_encoding = self.encode_climate_state(location, surface_vars, atmospheric_vars)
        
        # Solar resource (using month 6 as example - summer)
        month = torch.full((B,), 6, dtype=torch.long, device=device)
        solar_encoding = self.solar_encoder(location, surface_vars, month)
        
        # Wind resource
        wind_encoding = self.wind_encoder(location, surface_vars, roughness, hub_height)
        
        # Extreme event probabilities
        extreme_probs = self.extreme_predictor(climate_encoding)
        
        # Climate projections
        projections = self.climate_projector(climate_encoding, scenario, target_year)
        
        # Resource predictions
        ghi = self.ghi_predictor(solar_encoding)
        wind_speed = self.wind_speed_predictor(wind_encoding)
        
        # Aggregate risk score
        extreme_tensor = torch.stack(list(extreme_probs.values()), dim=-1)
        risk_input = torch.cat([climate_encoding, extreme_tensor], dim=-1)
        risk_score = self.risk_aggregator(risk_input).squeeze(-1)
        
        return {
            "risk_score": risk_score,
            "ghi_p10": ghi[:, 0],
            "ghi_p50": ghi[:, 1],
            "ghi_p90": ghi[:, 2],
            "wind_speed_p10": wind_speed[:, 0],
            "wind_speed_p50": wind_speed[:, 1],
            "wind_speed_p90": wind_speed[:, 2],
            "extreme_event_probs": extreme_probs,
            "climate_projections": projections,
            "climate_encoding": climate_encoding
        }
    
    @torch.no_grad()
    def assess_risk(
        self,
        latitude: float,
        longitude: float,
        elevation: float = 0,
        asset_type: str = "solar",
        scenario: ClimateScenario = ClimateScenario.SSP245,
        target_year: int = 2050,
        surface_vars: Optional[np.ndarray] = None
    ) -> ClimateRiskOutput:
        """
        Convenience method for single-location risk assessment.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            elevation: Elevation in meters
            asset_type: "solar" or "wind"
            scenario: Climate scenario
            target_year: Target year for projections
            surface_vars: Optional surface climate variables (uses defaults if None)
            
        Returns:
            ClimateRiskOutput dataclass
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Prepare inputs
        location = torch.tensor([[latitude, longitude, elevation]], device=device)
        
        if surface_vars is None:
            # Use placeholder values (in practice, fetch from ERA5 or similar)
            surface_vars = torch.tensor([[
                288.0,    # t2m: ~15°C
                0.002,    # tp: 2mm/day
                200.0,    # ssrd: 200 W/m²
                2.0,      # u10: 2 m/s
                1.0,      # v10: 1 m/s
                101325.0, # sp: 1 atm
                25.0      # tcwv: 25 kg/m²
            ]], device=device)
        else:
            surface_vars = torch.tensor([surface_vars], device=device)
        
        scenario_idx = torch.tensor([list(ClimateScenario).index(scenario)], device=device)
        target_year_t = torch.tensor([target_year], device=device)
        
        # Forward pass
        outputs = self.forward(
            location, 
            surface_vars, 
            scenario_idx, 
            target_year_t
        )
        
        # Convert to output format
        return ClimateRiskOutput(
            latitude=latitude,
            longitude=longitude,
            risk_score=outputs["risk_score"].item(),
            solar_ghi_kwh_m2_year={
                "p10": outputs["ghi_p10"].item() * 365,  # Daily to annual
                "p50": outputs["ghi_p50"].item() * 365,
                "p90": outputs["ghi_p90"].item() * 365
            },
            wind_speed_m_s={
                "p10": outputs["wind_speed_p10"].item(),
                "p50": outputs["wind_speed_p50"].item(),
                "p90": outputs["wind_speed_p90"].item()
            },
            capacity_factor_projection=[],  # Would need time series
            extreme_event_probs={
                k: v.item() for k, v in outputs["extreme_event_probs"].items()
            },
            temperature_change_c=outputs["climate_projections"]["temperature_delta_c"].item(),
            precipitation_change_pct=outputs["climate_projections"]["precipitation_delta_pct"].item(),
            confidence=0.8,  # Placeholder
            uncertainty_range=(0.0, 1.0)
        )


def create_climate_model(
    config: Optional[ClimateConfig] = None
) -> ClimateRiskModel:
    """Factory function to create climate risk model."""
    config = config or ClimateConfig()
    return ClimateRiskModel(config)


if __name__ == "__main__":
    print("Testing ClimateRiskModel...")
    
    config = ClimateConfig()
    model = ClimateRiskModel(config)
    
    # Test single assessment
    risk = model.assess_risk(
        latitude=35.0,
        longitude=-119.9,
        elevation=500,
        asset_type="solar",
        scenario=ClimateScenario.SSP245,
        target_year=2050
    )
    
    print(f"\nRisk Assessment for ({risk.latitude}, {risk.longitude}):")
    print(f"  Risk Score: {risk.risk_score:.3f}")
    print(f"  Solar GHI (P50): {risk.solar_ghi_kwh_m2_year['p50']:.0f} kWh/m²/year")
    print(f"  Wind Speed (P50): {risk.wind_speed_m_s['p50']:.1f} m/s")
    print(f"  Temperature Change: {risk.temperature_change_c:+.1f}°C")
    print(f"  Extreme Event Probabilities:")
    for event, prob in risk.extreme_event_probs.items():
        print(f"    {event}: {prob:.3f}")
