"""
General infrastructure suitability factors.

These factors apply to ALL energy types and evaluate site accessibility,
grid connectivity, and environmental/regulatory constraints.
"""

from src.factors.base import BaseFactor


class GridProximityFactor(BaseFactor):
    """Score proximity to electrical transmission infrastructure.

    Grid interconnection is one of the largest cost drivers for
    renewable projects. Sites closer to existing transmission lines
    or substations have lower interconnection costs and shorter
    permitting timelines.
    """

    def __init__(self, weight: float = 1.5, enabled: bool = True):
        super().__init__(
            name="Grid Proximity",
            description=(
                "Evaluates distance to nearest transmission line or substation. "
                "Closer = lower interconnection cost. Data from EIA (US) or "
                "OpenStreetMap power infrastructure (global)."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute grid proximity suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                grid_distance_km (float, optional): Distance to nearest
                    transmission line or substation in km.

        Returns:
            Normalized score 0-1.

        TODO:
            - Load EIA transmission line shapefiles (US)
            - Load OSM power=line and power=substation (global)
            - Compute nearest-feature distance with spatial index
            - Scoring (exponential decay):
                < 1 km: 1.0
                1-5 km: 0.85
                5-15 km: 0.6
                15-30 km: 0.35
                30-50 km: 0.15
                > 50 km: 0.05
        """
        dist = kwargs.get("grid_distance_km")
        if dist is not None:
            if dist < 1:
                return 1.0
            elif dist < 5:
                return 0.85
            elif dist < 15:
                return 0.6
            elif dist < 30:
                return 0.35
            elif dist < 50:
                return 0.15
            else:
                return 0.05
        return 0.5


class RoadAccessFactor(BaseFactor):
    """Score road access for construction and maintenance.

    Renewable energy projects require road access for equipment delivery
    (turbine blades, solar panels, transformers) and ongoing maintenance.
    Remote sites without road access incur significant additional cost.
    """

    def __init__(self, weight: float = 0.8, enabled: bool = True):
        super().__init__(
            name="Road Access",
            description=(
                "Evaluates road access for construction logistics. Proximity "
                "to paved roads reduces transport cost for heavy equipment. "
                "Data from OpenStreetMap highway network."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute road access suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                road_distance_km (float, optional): Distance to nearest
                    paved road (highway=primary/secondary/tertiary) in km.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query OSM highway=primary|secondary|tertiary within radius
            - Compute nearest-road distance
            - Scoring:
                < 0.5 km: 1.0
                0.5-2 km: 0.8
                2-5 km: 0.6
                5-15 km: 0.3
                > 15 km: 0.1
        """
        dist = kwargs.get("road_distance_km")
        if dist is not None:
            if dist < 0.5:
                return 1.0
            elif dist < 2:
                return 0.8
            elif dist < 5:
                return 0.6
            elif dist < 15:
                return 0.3
            else:
                return 0.1
        return 0.5


class FloodRiskFactor(BaseFactor):
    """Score flood risk at a location.

    Flooding can damage ground-mounted solar arrays, substation equipment,
    and access roads. High flood risk zones require elevated mounting or
    are excluded entirely.

    Note: this factor is INVERTED -- lower flood risk = higher score.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__(
            name="Flood Risk",
            description=(
                "Evaluates flood risk exposure. High flood risk zones "
                "are penalized. Uses FEMA flood maps (US) or global "
                "flood hazard models. Lower risk = higher score."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute flood risk suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                flood_zone (str, optional): FEMA flood zone designation.
                    "X" = minimal risk, "A"/"AE" = 100-year floodplain,
                    "V"/"VE" = coastal high hazard.
                flood_probability (float, optional): Annual flood probability
                    (0-1). Alternative to zone designation.

        Returns:
            Normalized score 0-1 (higher = safer).

        TODO:
            - Query FEMA NFHL (National Flood Hazard Layer) for US
            - Query Global Flood Database (Tellman et al.) for international
            - Map flood zones to scores:
                Zone X (minimal): 1.0
                Zone X500 (500-year): 0.7
                Zone A/AE (100-year): 0.2
                Zone V/VE (coastal hazard): 0.0
        """
        zone = kwargs.get("flood_zone")
        if zone is not None:
            zone = zone.upper().strip()
            if zone in ("X", "X500_SHADED"):
                return 0.9
            elif zone in ("X500",):
                return 0.7
            elif zone in ("A", "AE", "AO", "AH"):
                return 0.2
            elif zone in ("V", "VE"):
                return 0.0
            else:
                return 0.5

        prob = kwargs.get("flood_probability")
        if prob is not None:
            return max(0.0, min(1.0, 1.0 - prob))

        return 0.5


class ProtectedAreaFactor(BaseFactor):
    """Score proximity to and overlap with protected areas.

    National parks, wilderness areas, wildlife refuges, and other
    protected lands are typically excluded from energy development.
    Sites overlapping or very close to protected areas score low.

    Note: this factor is INVERTED -- no protection = higher score.
    """

    def __init__(self, weight: float = 1.2, enabled: bool = True):
        super().__init__(
            name="Protected Area",
            description=(
                "Evaluates overlap with or proximity to protected areas "
                "(national parks, wilderness, wildlife refuges). Overlapping "
                "sites are excluded. Data from WDPA (World Database on "
                "Protected Areas) and PAD-US."
            ),
            energy_types=["all"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute protected area suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                in_protected_area (bool, optional): Whether the site
                    falls within a protected area boundary.
                protected_area_distance_km (float, optional): Distance
                    to nearest protected area boundary in km.

        Returns:
            Normalized score 0-1 (higher = less constrained).

        TODO:
            - Load PAD-US (US) or WDPA (global) protected area polygons
            - Check point-in-polygon for site location
            - If inside: score = 0.0 (hard exclusion)
            - Buffer zone scoring:
                < 1 km from boundary: 0.3 (possible permitting issues)
                1-5 km: 0.7 (some constraints)
                > 5 km: 1.0 (no constraint)
        """
        in_protected = kwargs.get("in_protected_area")
        if in_protected is True:
            return 0.0

        dist = kwargs.get("protected_area_distance_km")
        if dist is not None:
            if dist < 1:
                return 0.3
            elif dist < 5:
                return 0.7
            else:
                return 1.0

        return 0.5
