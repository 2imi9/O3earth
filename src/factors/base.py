"""
Base class for all suitability factors.

Every factor in the OpenEnergy-Engine scoring pipeline inherits from
BaseFactor and implements a compute() method that returns a normalized
score between 0.0 (completely unsuitable) and 1.0 (ideal).
"""

from abc import ABC, abstractmethod
from typing import List


class BaseFactor(ABC):
    """Abstract base class for all suitability scoring factors.

    Attributes:
        name: Human-readable factor name (e.g. "Solar Irradiance").
        description: Brief explanation of what this factor measures and why
            it matters for site suitability.
        energy_types: List of energy types this factor applies to.
            Valid values: "solar", "wind", "hydro", "geothermal", "all".
            When set to ["all"], the factor is used for every energy type.
        weight: Default weight for this factor in the scoring engine.
            The engine normalizes weights across all active factors, so
            absolute values matter only relative to other factors.
        enabled: Whether this factor is active in scoring. Disabled factors
            are skipped entirely (not scored as 0).
    """

    def __init__(
        self,
        name: str,
        description: str,
        energy_types: List[str],
        weight: float = 1.0,
        enabled: bool = True,
    ):
        self.name = name
        self.description = description
        self.energy_types = energy_types
        self.weight = weight
        self.enabled = enabled

    @abstractmethod
    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute a suitability score for the given location.

        Args:
            lat: Latitude in decimal degrees (WGS84).
            lon: Longitude in decimal degrees (WGS84).
            **kwargs: Additional context (e.g. date_range, embeddings,
                elevation data, cached API responses).

        Returns:
            A float between 0.0 and 1.0 where:
                0.0 = completely unsuitable
                0.5 = neutral / unknown
                1.0 = ideal conditions
        """
        ...

    def applies_to(self, energy_type: str) -> bool:
        """Check whether this factor applies to the given energy type.

        Args:
            energy_type: One of "solar", "wind", "hydro", "geothermal".

        Returns:
            True if this factor should be included in scoring for
            the given energy type.
        """
        return "all" in self.energy_types or energy_type in self.energy_types

    def __repr__(self) -> str:
        status = "ON" if self.enabled else "OFF"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"weight={self.weight} [{status}]>"
        )
