"""
Suitability Scoring Engine.

The SuitabilityEngine combines configurable factors into a weighted
suitability score for a given energy type and location. It supports
enabling/disabling individual factors, adjusting weights, and batch
scoring across multiple locations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.factors.base import BaseFactor
from src.factors.land import (
    LandCoverFactor,
    TerrainElevationFactor,
    TerrainSlopeFactor,
)
from src.factors.solar import (
    CloudCoverageFactor,
    SolarIrradianceFactor,
    TemperatureEffectFactor,
)
from src.factors.wind import (
    TerrainRoughnessFactor,
    WindDirectionConsistencyFactor,
    WindSpeedFactor,
)
from src.factors.hydro import (
    ElevationDropFactor,
    WaterFlowFactor,
    WatershedHealthFactor,
)
from src.factors.geothermal import (
    FaultProximityFactor,
    HeatFlowFactor,
    SeismicActivityFactor,
)
from src.factors.infrastructure import (
    FloodRiskFactor,
    GridProximityFactor,
    ProtectedAreaFactor,
    RoadAccessFactor,
)
from src.scoring.weights import get_weights_for_energy_type

logger = logging.getLogger(__name__)

VALID_ENERGY_TYPES = {"solar", "wind", "hydro", "geothermal"}


@dataclass
class SuitabilityResult:
    """Result of a suitability scoring computation.

    Attributes:
        overall_score: Weighted average suitability score (0-1).
        factor_scores: Dict mapping factor name -> individual score (0-1).
        factor_weights: Dict mapping factor name -> weight used in scoring.
        energy_type: Energy type that was scored.
        lat: Latitude of scored location.
        lon: Longitude of scored location.
        timestamp: UTC timestamp of when scoring was performed.
        metadata: Optional dict for additional context (e.g. raw API
            responses, model version).
    """

    overall_score: float
    factor_scores: Dict[str, float]
    factor_weights: Dict[str, float]
    energy_type: str
    lat: float
    lon: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "overall_score": round(self.overall_score, 4),
            "factor_scores": {k: round(v, 4) for k, v in self.factor_scores.items()},
            "factor_weights": {k: round(v, 4) for k, v in self.factor_weights.items()},
            "energy_type": self.energy_type,
            "lat": self.lat,
            "lon": self.lon,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# Registry of all available factor classes.
# Each entry is instantiated when a SuitabilityEngine is created.
ALL_FACTOR_CLASSES: List[type] = [
    # Land (general)
    LandCoverFactor,
    TerrainSlopeFactor,
    TerrainElevationFactor,
    # Solar
    SolarIrradianceFactor,
    CloudCoverageFactor,
    TemperatureEffectFactor,
    # Wind
    WindSpeedFactor,
    WindDirectionConsistencyFactor,
    TerrainRoughnessFactor,
    # Hydro
    WaterFlowFactor,
    ElevationDropFactor,
    WatershedHealthFactor,
    # Geothermal
    HeatFlowFactor,
    FaultProximityFactor,
    SeismicActivityFactor,
    # Infrastructure (general)
    GridProximityFactor,
    RoadAccessFactor,
    FloodRiskFactor,
    ProtectedAreaFactor,
]


class SuitabilityEngine:
    """Main scoring engine that combines multiple factors into a site
    suitability score.

    Usage::

        engine = SuitabilityEngine("solar")
        result = engine.score(35.0, -120.0, ghi_kwh_m2_day=5.8)
        print(result.overall_score)  # 0.0 - 1.0

    The engine automatically selects factors that apply to the given
    energy type, applies configured weights, and computes a weighted
    average score.
    """

    def __init__(self, energy_type: str, custom_weights: Optional[Dict[str, float]] = None):
        """Initialize the scoring engine for a specific energy type.

        Args:
            energy_type: One of "solar", "wind", "hydro", "geothermal".
            custom_weights: Optional dict of factor_name -> weight overrides.
                If not provided, default weights from weights.py are used.

        Raises:
            ValueError: If energy_type is not recognized.
        """
        if energy_type not in VALID_ENERGY_TYPES:
            raise ValueError(
                f"Unknown energy type {energy_type!r}. "
                f"Valid types: {sorted(VALID_ENERGY_TYPES)}"
            )

        self.energy_type = energy_type
        self.factors: Dict[str, BaseFactor] = {}

        # Instantiate all factors and keep those that apply
        default_weights = get_weights_for_energy_type(energy_type)
        for cls in ALL_FACTOR_CLASSES:
            factor = cls()
            if factor.applies_to(energy_type):
                # Apply default weight overrides for this energy type
                if factor.name in default_weights:
                    factor.weight = default_weights[factor.name]
                self.factors[factor.name] = factor

        # Apply any custom weight overrides
        if custom_weights:
            for name, weight in custom_weights.items():
                if name in self.factors:
                    self.factors[name].weight = weight
                else:
                    logger.warning(
                        "Custom weight for unknown factor %r ignored", name
                    )

        logger.info(
            "SuitabilityEngine initialized for %s with %d factors",
            energy_type,
            len(self.factors),
        )

    def enable_factor(self, name: str) -> None:
        """Enable a factor by name.

        Args:
            name: Factor name (e.g. "Solar Irradiance").

        Raises:
            KeyError: If no factor with that name exists for this energy type.
        """
        if name not in self.factors:
            raise KeyError(
                f"Factor {name!r} not found. Available: {list(self.factors.keys())}"
            )
        self.factors[name].enabled = True

    def disable_factor(self, name: str) -> None:
        """Disable a factor by name. Disabled factors are skipped during scoring.

        Args:
            name: Factor name (e.g. "Flood Risk").

        Raises:
            KeyError: If no factor with that name exists for this energy type.
        """
        if name not in self.factors:
            raise KeyError(
                f"Factor {name!r} not found. Available: {list(self.factors.keys())}"
            )
        self.factors[name].enabled = False

    def set_weight(self, name: str, weight: float) -> None:
        """Set the weight for a specific factor.

        Args:
            name: Factor name.
            weight: New weight value (must be >= 0).

        Raises:
            KeyError: If no factor with that name exists.
            ValueError: If weight is negative.
        """
        if name not in self.factors:
            raise KeyError(
                f"Factor {name!r} not found. Available: {list(self.factors.keys())}"
            )
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        self.factors[name].weight = weight

    def get_active_factors(self) -> List[BaseFactor]:
        """Return list of currently enabled factors."""
        return [f for f in self.factors.values() if f.enabled]

    def score(self, lat: float, lon: float, **kwargs) -> SuitabilityResult:
        """Compute suitability score for a single location.

        The overall score is a weighted average of all enabled factors:
            score = sum(weight_i * score_i) / sum(weight_i)

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            **kwargs: Additional data passed to each factor's compute().
                Each factor picks out the kwargs it needs.

        Returns:
            SuitabilityResult with overall and per-factor scores.
        """
        active = self.get_active_factors()
        if not active:
            logger.warning("No active factors -- returning score 0.0")
            return SuitabilityResult(
                overall_score=0.0,
                factor_scores={},
                factor_weights={},
                energy_type=self.energy_type,
                lat=lat,
                lon=lon,
            )

        factor_scores: Dict[str, float] = {}
        factor_weights: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for factor in active:
            try:
                raw_score = factor.compute(lat, lon, **kwargs)
                # Clamp to [0, 1]
                clamped = max(0.0, min(1.0, raw_score))
                factor_scores[factor.name] = clamped
                factor_weights[factor.name] = factor.weight
                weighted_sum += factor.weight * clamped
                total_weight += factor.weight
            except Exception:
                logger.exception("Factor %r failed for (%s, %s)", factor.name, lat, lon)
                # Skip failed factors rather than crashing the whole score
                factor_scores[factor.name] = 0.0
                factor_weights[factor.name] = factor.weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return SuitabilityResult(
            overall_score=overall,
            factor_scores=factor_scores,
            factor_weights=factor_weights,
            energy_type=self.energy_type,
            lat=lat,
            lon=lon,
        )

    def batch_score(
        self, locations: List[Dict[str, Any]]
    ) -> List[SuitabilityResult]:
        """Score multiple locations.

        Args:
            locations: List of dicts, each containing at minimum "lat" and
                "lon" keys. Any additional keys are passed as kwargs to
                each factor's compute().

        Returns:
            List of SuitabilityResult, one per location, in the same order.

        Example::

            locations = [
                {"lat": 35.0, "lon": -120.0, "ghi_kwh_m2_day": 5.8},
                {"lat": 36.5, "lon": -118.0, "ghi_kwh_m2_day": 6.2},
            ]
            results = engine.batch_score(locations)
        """
        results = []
        for loc in locations:
            lat = loc["lat"]
            lon = loc["lon"]
            kwargs = {k: v for k, v in loc.items() if k not in ("lat", "lon")}

            result = self.score(lat, lon, **kwargs)
            results.append(result)
        return results

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of the engine configuration.

        Returns:
            Dict with energy_type, factors (name, weight, enabled, description).
        """
        return {
            "energy_type": self.energy_type,
            "total_factors": len(self.factors),
            "active_factors": len(self.get_active_factors()),
            "factors": [
                {
                    "name": f.name,
                    "weight": f.weight,
                    "enabled": f.enabled,
                    "description": f.description,
                    "energy_types": f.energy_types,
                }
                for f in self.factors.values()
            ],
        }

    def __repr__(self) -> str:
        active = len(self.get_active_factors())
        total = len(self.factors)
        return (
            f"<SuitabilityEngine energy_type={self.energy_type!r} "
            f"factors={active}/{total} active>"
        )
