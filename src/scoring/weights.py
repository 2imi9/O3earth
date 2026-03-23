"""
Default weight configurations per energy type.

Weights control how much each factor contributes to the overall
suitability score. The engine normalizes weights (divides by sum)
so only relative magnitudes matter.

Weight design rationale:
- Primary resource factors (irradiance, wind speed, water flow, heat flow)
  get the highest weights (~2.0) because no resource = no project.
- Land characterization and grid proximity are important for all types (~1.5).
- Secondary factors (temperature, roughness, road access) get moderate
  weights (~0.8-1.0).
- Constraint factors (flood risk, protected areas) act more as filters
  and get weights of ~1.0-1.2.
"""

from typing import Dict


# Default weights keyed by factor name.
# These are applied when SuitabilityEngine is initialized for a given
# energy type. Factors not listed here use their class-level defaults.

SOLAR_WEIGHTS: Dict[str, float] = {
    # Primary resource
    "Solar Irradiance": 2.5,
    "Cloud Coverage": 1.2,
    "Temperature Effect": 0.6,
    # Land / terrain
    "Land Cover": 1.5,
    "Terrain Slope": 1.2,       # Solar needs flat land
    "Terrain Elevation": 0.4,
    # Infrastructure
    "Grid Proximity": 1.5,
    "Road Access": 0.8,
    "Flood Risk": 1.0,
    "Protected Area": 1.2,
}

WIND_WEIGHTS: Dict[str, float] = {
    # Primary resource
    "Wind Speed": 2.5,
    "Wind Direction Consistency": 1.0,
    "Terrain Roughness": 1.2,
    # Land / terrain
    "Land Cover": 1.0,
    "Terrain Slope": 0.6,       # Wind tolerates slopes better
    "Terrain Elevation": 0.5,
    # Infrastructure
    "Grid Proximity": 1.5,
    "Road Access": 1.0,          # Heavy equipment (blades)
    "Flood Risk": 0.8,
    "Protected Area": 1.2,
}

HYDRO_WEIGHTS: Dict[str, float] = {
    # Primary resource
    "Water Flow": 2.5,
    "Elevation Drop": 2.0,
    "Watershed Health": 1.0,
    # Land / terrain
    "Land Cover": 0.8,
    "Terrain Slope": 0.3,       # Hydro sites are inherently sloped
    "Terrain Elevation": 0.3,
    # Infrastructure
    "Grid Proximity": 1.5,
    "Road Access": 0.8,
    "Flood Risk": 0.5,           # Hydro is inherently near water
    "Protected Area": 1.5,       # Dams face heavy environmental review
}

GEOTHERMAL_WEIGHTS: Dict[str, float] = {
    # Primary resource
    "Heat Flow": 2.5,
    "Fault Proximity": 1.8,
    "Seismic Activity": 1.2,
    # Land / terrain
    "Land Cover": 0.8,
    "Terrain Slope": 0.5,
    "Terrain Elevation": 0.4,
    # Infrastructure
    "Grid Proximity": 1.5,
    "Road Access": 1.0,          # Drilling rigs need road access
    "Flood Risk": 0.8,
    "Protected Area": 1.2,
}

DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "solar": SOLAR_WEIGHTS,
    "wind": WIND_WEIGHTS,
    "hydro": HYDRO_WEIGHTS,
    "geothermal": GEOTHERMAL_WEIGHTS,
}


def get_weights_for_energy_type(energy_type: str) -> Dict[str, float]:
    """Return the default weight configuration for an energy type.

    Args:
        energy_type: One of "solar", "wind", "hydro", "geothermal".

    Returns:
        Dict mapping factor name -> default weight.

    Raises:
        KeyError: If energy_type is not recognized.
    """
    if energy_type not in DEFAULT_WEIGHTS:
        raise KeyError(
            f"No default weights for {energy_type!r}. "
            f"Available: {sorted(DEFAULT_WEIGHTS.keys())}"
        )
    return DEFAULT_WEIGHTS[energy_type].copy()
