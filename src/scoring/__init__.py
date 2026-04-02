"""
OpenEnergy-Engine Scoring Module.

Provides the SuitabilityEngine that combines multiple factors into
a single site suitability score per energy type.
"""

from src.scoring.engine import SuitabilityEngine, SuitabilityResult
from src.scoring.weights import DEFAULT_WEIGHTS, get_weights_for_energy_type

__all__ = [
    "SuitabilityEngine",
    "SuitabilityResult",
    "DEFAULT_WEIGHTS",
    "get_weights_for_energy_type",
]
