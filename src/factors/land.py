"""
Land characterization factors derived from OlmoEarth embeddings and
elevation data.

These factors apply to ALL energy types because every renewable project
needs buildable land with suitable terrain.
"""

from src.factors.base import BaseFactor


class LandCoverFactor(BaseFactor):
    """Score land suitability using OlmoEarth 768-dim embeddings.

    OlmoEarth encodes land cover gradients, vegetation density, built-up
    area proximity, and surface texture into a continuous embedding space.
    This factor projects those embeddings into a suitability score.

    High scores: barren/scrubland, low-value agricultural, brownfield
    Low scores: dense forest, wetland, urban core, water body
    """

    def __init__(self, weight: float = 1.5, enabled: bool = True):
        super().__init__(
            name="Land Cover",
            description=(
                "Uses OlmoEarth embeddings to assess land cover suitability. "
                "Favors barren, scrubland, and low-value agricultural land. "
                "Penalizes dense forest, wetland, urban cores, and water bodies."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute land cover suitability from OlmoEarth embeddings.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                embeddings (np.ndarray, optional): Pre-computed OlmoEarth
                    768-dim embedding vector for this location.

        Returns:
            Normalized score 0-1.

        TODO:
            - Load OlmoEarth BASE checkpoint from
              project_data/openenergyengine/run_21_frozen/best.ckpt
            - Fetch Sentinel-2 patch via data_clients.satellite
            - Run forward pass to get 768-dim embedding
            - Train a linear probe on EIA 860 plant locations to map
              embeddings -> suitability score
            - Cache embeddings per (lat, lon, date) to avoid recomputation
        """
        embeddings = kwargs.get("embeddings")
        if embeddings is not None:
            # TODO: Apply trained linear probe to embeddings
            # For now return neutral score
            return 0.5
        return 0.5


class TerrainSlopeFactor(BaseFactor):
    """Score terrain slope suitability.

    Solar farms need flat land (< 5 degrees ideal).
    Wind farms tolerate moderate slopes but need ridge exposure.
    Hydro needs elevation drop (handled separately).
    Geothermal is slope-tolerant.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__(
            name="Terrain Slope",
            description=(
                "Evaluates terrain slope from elevation data. "
                "Solar favors flat land (< 5 deg). Wind tolerates moderate slopes. "
                "Steep terrain penalized for construction difficulty."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute slope suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                slope_degrees (float, optional): Pre-computed slope in degrees.
                energy_type (str, optional): Used to adjust thresholds.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query USGS 3DEP (US) or SRTM (global) for elevation tile
            - Compute slope from DEM using numpy gradient
            - Apply energy-type-specific thresholds:
                Solar: score = max(0, 1 - slope/15)
                Wind: score = 1.0 if 2 < slope < 20, else penalize
                Hydro/Geothermal: more tolerant
        """
        slope = kwargs.get("slope_degrees")
        if slope is not None:
            # Simple linear decay: ideal at 0 deg, unsuitable at 15+ deg
            return max(0.0, min(1.0, 1.0 - slope / 15.0))
        return 0.5


class TerrainElevationFactor(BaseFactor):
    """Score elevation suitability.

    Extremely high elevations increase construction cost and reduce
    equipment efficiency. Very low elevations may indicate flood risk.
    """

    def __init__(self, weight: float = 0.5, enabled: bool = True):
        super().__init__(
            name="Terrain Elevation",
            description=(
                "Evaluates elevation suitability. Very high elevations increase "
                "construction costs; very low elevations may indicate flood risk. "
                "Moderate elevations (100-2000m) are generally preferred."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute elevation suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                elevation_m (float, optional): Elevation in meters above
                    sea level.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query USGS 3DEP / SRTM for elevation at (lat, lon)
            - Apply scoring curve: penalize < 10m (flood risk) and
              > 3000m (construction difficulty, thin air)
            - Ideal range: 100-2000m -> score 1.0
        """
        elev = kwargs.get("elevation_m")
        if elev is not None:
            if elev < 0:
                return 0.0
            elif elev < 10:
                return 0.3  # Flood risk zone
            elif elev < 100:
                return 0.7
            elif elev <= 2000:
                return 1.0
            elif elev <= 3000:
                return 0.6
            else:
                return 0.3
        return 0.5
