"""
Wind-specific suitability factors.

These factors evaluate conditions for onshore wind farm installations:
wind speed at hub height, directional consistency, and terrain roughness.
"""

from src.factors.base import BaseFactor


class WindSpeedFactor(BaseFactor):
    """Score wind resource using average wind speed at hub height.

    Modern utility-scale turbines (hub height ~80-120m) need sustained
    wind speeds of 6-9 m/s for economic viability.  Class 3+ wind
    resource (> 6.4 m/s at 80m) is the typical development threshold.
    """

    def __init__(self, weight: float = 2.0, enabled: bool = True):
        super().__init__(
            name="Wind Speed",
            description=(
                "Evaluates average wind speed at hub height (80-120m). "
                "Data from NREL Wind Toolkit (US) or Global Wind Atlas. "
                "Class 3+ wind (> 6.4 m/s) required for economic viability."
            ),
            energy_types=["wind"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute wind speed suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                wind_speed_ms (float, optional): Annual mean wind speed
                    at hub height in m/s.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query NREL Wind Toolkit API for US locations
            - Query Global Wind Atlas for international locations
            - Use 80m or 100m hub height wind speed
            - Scoring: < 4 m/s -> 0.1, 4-5 -> 0.3, 5-6 -> 0.5,
              6-7 -> 0.7, 7-9 -> 0.9, > 9 -> 1.0
        """
        speed = kwargs.get("wind_speed_ms")
        if speed is None:
            # Estimate wind from latitude (coastal/plains heuristic)
            abs_lat = abs(lat)
            if 35 <= abs_lat <= 55:
                speed = 7.0  # Westerlies belt
            elif 25 <= abs_lat < 35:
                speed = 5.5  # Subtropics
            elif abs_lat < 25:
                speed = 4.5  # Tropics (calmer)
            else:
                speed = 6.0  # Polar regions

        if speed < 3.0:
            return 0.0
        elif speed < 4.0:
            return 0.1
        elif speed < 5.0:
            return 0.3
        elif speed < 6.0:
            return 0.5
        elif speed < 7.0:
            return 0.7
        elif speed < 9.0:
            return 0.9
        else:
            return 1.0


class WindDirectionConsistencyFactor(BaseFactor):
    """Score wind direction consistency (low variability = better).

    Sites where wind blows consistently from one direction allow optimal
    turbine layout. High directional variability reduces capacity factor
    and complicates wake management.
    """

    def __init__(self, weight: float = 0.8, enabled: bool = True):
        super().__init__(
            name="Wind Direction Consistency",
            description=(
                "Evaluates wind directional consistency. Consistent wind "
                "direction enables optimal turbine layout and reduces wake "
                "losses. Measured as concentration of wind rose."
            ),
            energy_types=["wind"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute wind direction consistency score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                direction_consistency (float, optional): Wind direction
                    consistency index (0 = omnidirectional, 1 = unidirectional).
                    Computed as the magnitude of the mean wind vector divided
                    by the mean wind speed.

        Returns:
            Normalized score 0-1.

        TODO:
            - Compute from hourly wind data (NREL Wind Toolkit / ERA5)
            - Calculate vector mean of wind direction over a year
            - Consistency = |mean_vector| / mean_speed
            - Score directly maps to consistency index
        """
        consistency = kwargs.get("direction_consistency")
        if consistency is not None:
            return max(0.0, min(1.0, consistency))
        return 0.5


class TerrainRoughnessFactor(BaseFactor):
    """Score terrain roughness for wind energy.

    Surface roughness affects wind speed profile and turbulence. Open
    flat terrain (grassland, water) has low roughness and better wind.
    Forests and urban areas create turbulence.

    Note: this differs from TerrainSlopeFactor (which measures gradient).
    Roughness measures surface texture at a landscape scale.
    """

    def __init__(self, weight: float = 0.8, enabled: bool = True):
        super().__init__(
            name="Terrain Roughness",
            description=(
                "Evaluates surface roughness affecting wind profile. Low roughness "
                "(open grassland, water) allows higher wind speeds at hub height. "
                "High roughness (forest, urban) creates turbulence. "
                "Derived from DEM texture analysis or land cover classification."
            ),
            energy_types=["wind"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute terrain roughness suitability for wind energy.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                roughness_length_m (float, optional): Aerodynamic roughness
                    length in meters. Open water ~0.0002, grassland ~0.03,
                    forest ~1.0, urban ~2.0.

        Returns:
            Normalized score 0-1.

        TODO:
            - Derive roughness from SRTM DEM texture analysis
            - Alternative: map NLCD/OlmoEarth land cover to roughness class
            - Standard roughness classes:
                0: water (z0 ~ 0.0002m) -> score 1.0
                I: open (z0 ~ 0.01m) -> score 0.9
                II: grassland (z0 ~ 0.05m) -> score 0.8
                III: suburbs (z0 ~ 0.3m) -> score 0.4
                IV: urban (z0 ~ 1.0m) -> score 0.1
        """
        z0 = kwargs.get("roughness_length_m")
        if z0 is not None:
            if z0 <= 0.001:
                return 1.0
            elif z0 <= 0.03:
                return 0.9
            elif z0 <= 0.1:
                return 0.8
            elif z0 <= 0.3:
                return 0.5
            elif z0 <= 1.0:
                return 0.2
            else:
                return 0.1
        return 0.5
