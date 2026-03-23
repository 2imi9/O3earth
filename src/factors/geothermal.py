"""
Geothermal-specific suitability factors.

These factors evaluate conditions for geothermal energy installations:
subsurface heat flow, proximity to fault systems, and seismic activity
(which correlates with geothermal potential).
"""

from src.factors.base import BaseFactor


class HeatFlowFactor(BaseFactor):
    """Score subsurface heat flow density.

    Geothermal viability depends on accessible heat. The continental
    average is ~65 mW/m2.  Viable geothermal sites typically have
    > 80 mW/m2, with enhanced geothermal systems (EGS) possible at
    > 150 mW/m2.
    """

    def __init__(self, weight: float = 2.0, enabled: bool = True):
        super().__init__(
            name="Heat Flow",
            description=(
                "Evaluates subsurface heat flow density (mW/m2). "
                "Continental average ~65 mW/m2. Viable geothermal > 80 mW/m2. "
                "Data from USGS heat flow database and global compilations."
            ),
            energy_types=["geothermal"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute heat flow suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                heat_flow_mwm2 (float, optional): Heat flow density
                    in milliwatts per square meter.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query USGS heat flow database (US)
            - Query Global Heat Flow Database for international
            - Interpolate between measurement points (IDW or kriging)
            - Scoring:
                < 40 mW/m2: 0.0 (cold crust)
                40-65: 0.2 (below average)
                65-80: 0.4 (average, marginal)
                80-120: 0.7 (good potential)
                120-200: 0.9 (excellent)
                > 200: 1.0 (hotspot)
        """
        hf = kwargs.get("heat_flow_mwm2")
        if hf is not None:
            if hf < 40:
                return 0.0
            elif hf < 65:
                return 0.2
            elif hf < 80:
                return 0.4
            elif hf < 120:
                return 0.7
            elif hf < 200:
                return 0.9
            else:
                return 1.0
        return 0.5


class FaultProximityFactor(BaseFactor):
    """Score proximity to geological fault systems.

    Active fault zones concentrate geothermal fluids and heat.
    Closer to faults = higher geothermal potential, but also
    higher seismic risk (handled separately).
    """

    def __init__(self, weight: float = 1.5, enabled: bool = True):
        super().__init__(
            name="Fault Proximity",
            description=(
                "Evaluates proximity to mapped geological faults. Active "
                "fault zones concentrate geothermal fluids. Closer to faults "
                "= higher potential. Data from USGS Quaternary Fault Database."
            ),
            energy_types=["geothermal"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute fault proximity suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                fault_distance_km (float, optional): Distance to nearest
                    mapped fault in kilometers.

        Returns:
            Normalized score 0-1.

        TODO:
            - Load USGS Quaternary Fault and Fold Database shapefile
            - Compute nearest-fault distance using spatial index
            - Scoring (distance decay):
                < 1 km: 1.0 (on/near fault)
                1-5 km: 0.8
                5-15 km: 0.5
                15-50 km: 0.2
                > 50 km: 0.05
        """
        dist = kwargs.get("fault_distance_km")
        if dist is not None:
            if dist < 1:
                return 1.0
            elif dist < 5:
                return 0.8
            elif dist < 15:
                return 0.5
            elif dist < 50:
                return 0.2
            else:
                return 0.05
        return 0.5


class SeismicActivityFactor(BaseFactor):
    """Score seismic activity as a geothermal indicator.

    For geothermal, moderate seismicity is POSITIVE (indicates active
    tectonics and heat flow). Very high seismicity may increase
    infrastructure risk but still indicates resource presence.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__(
            name="Seismic Activity",
            description=(
                "Evaluates seismic activity as a geothermal resource indicator. "
                "Moderate seismicity correlates with active heat flow. "
                "Data from USGS earthquake catalog (event density within radius)."
            ),
            energy_types=["geothermal"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute seismic activity score for geothermal suitability.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                earthquake_density (float, optional): Number of M2.0+
                    earthquakes per 10,000 km2 per year within 50km radius.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query USGS Earthquake Hazards API for events within 50km
              radius over last 10 years
            - Compute event density (events per area per year)
            - Scoring (inverted U -- moderate is best):
                0 events: 0.2 (tectonically dead)
                1-5: 0.5 (low activity)
                5-20: 0.8 (moderate, good indicator)
                20-50: 1.0 (active, strong indicator)
                > 50: 0.7 (very active, infrastructure risk)
        """
        density = kwargs.get("earthquake_density")
        if density is not None:
            if density <= 0:
                return 0.2
            elif density <= 5:
                return 0.5
            elif density <= 20:
                return 0.8
            elif density <= 50:
                return 1.0
            else:
                return 0.7  # High seismicity: good resource, some risk
        return 0.5
