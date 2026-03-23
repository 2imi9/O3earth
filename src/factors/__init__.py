"""
OpenEnergy-Engine Factor Module.

Exports all factor classes used by the suitability scoring engine.
Each factor computes a normalized 0-1 score for a given location.
"""

from src.factors.base import BaseFactor

# Land characterization factors (OlmoEarth-derived)
from src.factors.land import (
    LandCoverFactor,
    TerrainSlopeFactor,
    TerrainElevationFactor,
)

# Solar-specific factors
from src.factors.solar import (
    SolarIrradianceFactor,
    CloudCoverageFactor,
    TemperatureEffectFactor,
)

# Wind-specific factors
from src.factors.wind import (
    WindSpeedFactor,
    WindDirectionConsistencyFactor,
    TerrainRoughnessFactor,
)

# Hydro-specific factors
from src.factors.hydro import (
    WaterFlowFactor,
    ElevationDropFactor,
    WatershedHealthFactor,
)

# Geothermal-specific factors
from src.factors.geothermal import (
    HeatFlowFactor,
    FaultProximityFactor,
    SeismicActivityFactor,
)

# General infrastructure factors
from src.factors.infrastructure import (
    GridProximityFactor,
    RoadAccessFactor,
    FloodRiskFactor,
    ProtectedAreaFactor,
)

__all__ = [
    "BaseFactor",
    # Land
    "LandCoverFactor",
    "TerrainSlopeFactor",
    "TerrainElevationFactor",
    # Solar
    "SolarIrradianceFactor",
    "CloudCoverageFactor",
    "TemperatureEffectFactor",
    # Wind
    "WindSpeedFactor",
    "WindDirectionConsistencyFactor",
    "TerrainRoughnessFactor",
    # Hydro
    "WaterFlowFactor",
    "ElevationDropFactor",
    "WatershedHealthFactor",
    # Geothermal
    "HeatFlowFactor",
    "FaultProximityFactor",
    "SeismicActivityFactor",
    # Infrastructure
    "GridProximityFactor",
    "RoadAccessFactor",
    "FloodRiskFactor",
    "ProtectedAreaFactor",
]
