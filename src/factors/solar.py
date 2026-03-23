"""
Solar-specific suitability factors.

These factors evaluate conditions specific to solar PV and CSP installations:
irradiance (GHI/DNI), cloud coverage, and temperature effects on panel
efficiency.
"""

from src.factors.base import BaseFactor


class SolarIrradianceFactor(BaseFactor):
    """Score solar resource availability using GHI/DNI data.

    Global Horizontal Irradiance (GHI) is the primary metric for PV.
    Direct Normal Irradiance (DNI) matters more for CSP (concentrated solar).
    Typical good values: GHI > 5.0 kWh/m2/day, DNI > 6.0 kWh/m2/day.
    """

    def __init__(self, weight: float = 2.0, enabled: bool = True):
        super().__init__(
            name="Solar Irradiance",
            description=(
                "Evaluates solar resource using GHI (Global Horizontal Irradiance) "
                "and DNI (Direct Normal Irradiance). Data from NREL NSRDB or "
                "Global Solar Atlas. Higher irradiance = higher energy yield."
            ),
            energy_types=["solar"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute solar irradiance suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                ghi_kwh_m2_day (float, optional): Annual average GHI in
                    kWh/m2/day.
                dni_kwh_m2_day (float, optional): Annual average DNI in
                    kWh/m2/day.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query NREL NSRDB API for US locations
            - Query Global Solar Atlas for international locations
            - Use annual average GHI as primary metric
            - Scoring: GHI < 3.0 -> 0.2, 3.0-4.0 -> 0.4, 4.0-5.0 -> 0.6,
              5.0-6.0 -> 0.8, > 6.0 -> 1.0
        """
        ghi = kwargs.get("ghi_kwh_m2_day")
        if ghi is not None:
            if ghi <= 2.0:
                return 0.1
            elif ghi <= 3.0:
                return 0.3
            elif ghi <= 4.0:
                return 0.5
            elif ghi <= 5.0:
                return 0.7
            elif ghi <= 6.0:
                return 0.85
            else:
                return 1.0
        return 0.5


class CloudCoverageFactor(BaseFactor):
    """Score cloud coverage frequency at a location.

    Persistent cloud cover reduces solar yield significantly. This factor
    uses ERA5 reanalysis or Sentinel-2 cloud mask statistics.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__(
            name="Cloud Coverage",
            description=(
                "Evaluates annual cloud coverage frequency. Persistent clouds "
                "reduce solar yield. Uses ERA5 reanalysis or Sentinel-2 cloud "
                "mask statistics. Lower cloud fraction = higher score."
            ),
            energy_types=["solar"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute cloud coverage suitability score.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                cloud_fraction (float, optional): Annual mean cloud fraction
                    (0.0 = always clear, 1.0 = always cloudy).

        Returns:
            Normalized score 0-1.

        TODO:
            - Query ERA5 total cloud cover (TCC) for annual mean
            - Alternative: compute cloud fraction from Sentinel-2 SCL band
              across multiple acquisitions
            - Score = 1 - cloud_fraction (linear inversion)
        """
        cloud = kwargs.get("cloud_fraction")
        if cloud is not None:
            return max(0.0, min(1.0, 1.0 - cloud))
        return 0.5


class TemperatureEffectFactor(BaseFactor):
    """Score temperature impact on solar panel efficiency.

    Solar PV panels lose ~0.4% efficiency per degree C above 25C.
    Extremely cold climates can cause snow coverage issues.
    Ideal operating temperature: 15-25C ambient.
    """

    def __init__(self, weight: float = 0.5, enabled: bool = True):
        super().__init__(
            name="Temperature Effect",
            description=(
                "Evaluates temperature impact on PV efficiency. Panels lose "
                "~0.4%/C above 25C. Very cold climates risk snow coverage. "
                "Ideal ambient: 15-25C."
            ),
            energy_types=["solar"],
            weight=weight,
            enabled=enabled,
        )

    def compute(self, lat: float, lon: float, **kwargs) -> float:
        """Compute temperature effect on solar suitability.

        Args:
            lat: Latitude (WGS84).
            lon: Longitude (WGS84).
            **kwargs:
                avg_temp_c (float, optional): Annual average temperature
                    in degrees Celsius.

        Returns:
            Normalized score 0-1.

        TODO:
            - Query ERA5 2m temperature for annual average
            - Apply efficiency curve:
                < -10C: 0.3 (snow/ice risk)
                -10 to 10C: 0.6 (cold but viable)
                10 to 25C: 1.0 (ideal)
                25 to 35C: 0.8 (efficiency loss starts)
                35 to 45C: 0.5 (significant loss)
                > 45C: 0.3 (extreme heat)
        """
        temp = kwargs.get("avg_temp_c")
        if temp is not None:
            if temp < -10:
                return 0.3
            elif temp < 10:
                return 0.6 + 0.4 * (temp + 10) / 20  # Linear ramp 0.6 -> 1.0
            elif temp <= 25:
                return 1.0
            elif temp <= 35:
                return 1.0 - 0.2 * (temp - 25) / 10  # 1.0 -> 0.8
            elif temp <= 45:
                return 0.8 - 0.3 * (temp - 35) / 10  # 0.8 -> 0.5
            else:
                return 0.3
        return 0.5
