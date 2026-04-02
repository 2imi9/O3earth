"""
Hydro-specific suitability factors.

These factors evaluate conditions for hydroelectric installations:
water flow rate, elevation drop (head), and watershed health.
"""

from src.factors.base import BaseFactor


class WaterFlowFactor(BaseFactor):
    """Score water source flow rate and stability.

    Hydropower requires consistent water flow. High seasonal variability
    (monsoon-dominated or snowmelt-only) reduces capacity factor.
    Data from HydroSHEDS, GRDC river discharge records.
    """

    def __init__(self, weight: float = 2.0, enabled: bool = True):
        super().__init__(
            name="Water Flow",
            description=(
                "Evaluates river/stream flow rate and seasonal stability. "
                "Consistent year-round flow is ideal. Data from HydroSHEDS "
                "and GRDC global river discharge database."
            ),
            energy_types=["hydro"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute water flow suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                discharge_m3s (float, optional): Mean annual river discharge
                    in cubic meters per second.
                flow_variability (float, optional): Coefficient of variation
                    of monthly discharge (0 = constant, higher = more variable).

        Returns:
            Normalized score 0-1.

        TODO:
            - Query HydroSHEDS flow accumulation raster for nearest stream
            - Query GRDC for discharge measurements at nearest gauge
            - Score based on: (1) sufficient discharge for turbine sizing,
              (2) low seasonal variability
            - Combine: score = 0.6 * discharge_score + 0.4 * stability_score
        """
        discharge = kwargs.get("discharge_m3s")
        if discharge is not None:
            if discharge <= 0:
                return 0.0
            elif discharge < 1:
                return 0.2  # Very small stream
            elif discharge < 10:
                return 0.5  # Small river, micro-hydro viable
            elif discharge < 100:
                return 0.8  # Medium river, good potential
            else:
                return 1.0  # Large river, excellent potential
        return 0.5


class ElevationDropFactor(BaseFactor):
    """Score elevation drop (hydraulic head) at a location.

    Hydropower output is proportional to head (elevation drop) times
    flow rate: P = rho * g * Q * H * efficiency.  Greater head means
    more power per unit of water.
    """

    def __init__(self, weight: float = 1.5, enabled: bool = True):
        super().__init__(
            name="Elevation Drop",
            description=(
                "Evaluates hydraulic head (elevation drop) available for "
                "hydropower. Greater head = more power per unit flow. "
                "Derived from SRTM DEM along stream network."
            ),
            energy_types=["hydro"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute elevation drop suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                head_m (float, optional): Available hydraulic head in meters
                    over a reasonable distance (e.g. 1km stream reach).

        Returns:
            Normalized score 0-1.

        TODO:
            - Compute from SRTM DEM: find nearest stream pixel (from
              HydroSHEDS), trace upstream and downstream, measure drop
            - Scoring thresholds:
                < 2m: 0.1 (run-of-river barely viable)
                2-10m: 0.4 (low head)
                10-50m: 0.7 (medium head)
                50-200m: 0.9 (high head)
                > 200m: 1.0 (excellent head)
        """
        head = kwargs.get("head_m")
        if head is not None:
            if head < 2:
                return 0.1
            elif head < 10:
                return 0.4
            elif head < 50:
                return 0.7
            elif head < 200:
                return 0.9
            else:
                return 1.0
        return 0.5


class WatershedHealthFactor(BaseFactor):
    """Score watershed ecosystem health.

    Degraded watersheds have higher sediment loads (turbine damage),
    less predictable flows, and higher environmental permitting risk.
    Healthy vegetation in the watershed indicates stable hydrology.
    """

    def __init__(self, weight: float = 0.8, enabled: bool = True):
        super().__init__(
            name="Watershed Health",
            description=(
                "Evaluates upstream watershed vegetation health as a proxy "
                "for hydrological stability and sediment risk. Derived from "
                "Sentinel-2 NDVI in contributing watershed area."
            ),
            energy_types=["hydro"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute watershed health score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                watershed_ndvi (float, optional): Mean NDVI of the
                    contributing watershed area (0 to 1 range).

        Returns:
            Normalized score 0-1.

        TODO:
            - Delineate upstream watershed using HydroSHEDS flow direction
            - Compute mean NDVI from Sentinel-2 within watershed boundary
            - NDVI > 0.6: healthy (score 1.0)
            - NDVI 0.3-0.6: moderate (score 0.6)
            - NDVI < 0.3: degraded (score 0.2)
        """
        ndvi = kwargs.get("watershed_ndvi")
        if ndvi is not None:
            if ndvi >= 0.6:
                return 1.0
            elif ndvi >= 0.3:
                return 0.4 + (ndvi - 0.3) / 0.3 * 0.6
            else:
                return 0.2

        # Fallback: use precipitation as proxy for watershed health
        # Higher precipitation = more vegetation = healthier watershed
        precip = kwargs.get("precipitation_mm_day")
        if precip is not None:
            if precip >= 5.0:
                return 0.95  # Tropical/wet — healthy watershed
            elif precip >= 3.0:
                return 0.8
            elif precip >= 1.5:
                return 0.6
            elif precip >= 0.5:
                return 0.4
            else:
                return 0.2  # Arid — poor watershed

        return 0.5
