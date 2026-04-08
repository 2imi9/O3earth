"""
cBottle Data Client — km-scale climate fields from NVIDIA's cBottle model.

Provides high-resolution (~7km) climate data as an optional upgrade over
NASA POWER (~50km) for the factor engine. Requires GPU and cBottle/earth2studio
installed.

Variables extracted:
- rsds: Downward shortwave radiation (surface) → solar irradiance
- uas/vas: 10m wind components → wind speed
- tas: 2m temperature → temperature effect
- pr: Precipitation rate → flood risk / watershed
- pres_msl: Mean sea level pressure

Usage:
    from src.data_clients.cbottle import CBottleClient

    client = CBottleClient()
    if client.available:
        data = client.fetch(lat=35.0, lon=-120.0)
        # Returns dict compatible with factor engine kwargs

Reference: https://github.com/NVlabs/cBottle
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Track availability without hard dependency
_CBOTTLE_AVAILABLE = False
_EARTH2STUDIO_AVAILABLE = False

try:
    from earth2studio.data import CBottle3D
    _EARTH2STUDIO_AVAILABLE = True
    _CBOTTLE_AVAILABLE = True
except ImportError:
    pass

if not _EARTH2STUDIO_AVAILABLE:
    try:
        import cbottle.inference
        _CBOTTLE_AVAILABLE = True
    except ImportError:
        pass


class CBottleClient:
    """Client for extracting km-scale climate data from cBottle.

    Falls back gracefully when cBottle is not installed or no GPU is available.
    Designed to slot into the existing factor engine as a drop-in replacement
    for NASA POWER data with higher spatial resolution.

    The client caches the model after first load to avoid repeated initialization.
    """

    def __init__(self):
        self._model = None
        self._loaded = False

    @property
    def available(self) -> bool:
        """Check if cBottle can be used (installed + GPU)."""
        if not _CBOTTLE_AVAILABLE:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load the cBottle model on first use."""
        if self._loaded:
            return

        if _EARTH2STUDIO_AVAILABLE:
            try:
                package = CBottle3D.load_default_package()
                self._model = CBottle3D.load_model(package)
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                self._loaded = True
                logger.info("cBottle loaded via Earth2Studio")
            except Exception:
                logger.warning("Failed to load cBottle via Earth2Studio", exc_info=True)
        elif _CBOTTLE_AVAILABLE:
            try:
                import cbottle.inference
                self._model = cbottle.inference.load("cbottle-3d-moe")
                self._loaded = True
                logger.info("cBottle loaded via native API")
            except Exception:
                logger.warning("Failed to load cBottle via native API", exc_info=True)

    def fetch(
        self,
        lat: float,
        lon: float,
        date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch km-scale climate data for a location.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            date: Date for the climate snapshot. Defaults to recent climatology.

        Returns:
            Dict compatible with factor engine kwargs, or None if unavailable.
            Keys match NASA POWER output for drop-in replacement:
            - ghi_kwh_m2_day: Solar irradiance (from rsds)
            - wind_speed_ms: Wind speed at hub height (from uas/vas)
            - avg_temp_c: Temperature (from tas)
            - precipitation_mm_day: Precipitation (from pr)
        """
        if not self.available:
            return None

        self._load_model()
        if not self._loaded:
            return None

        if date is None:
            date = datetime(2022, 6, 15)  # Recent summer baseline

        try:
            if _EARTH2STUDIO_AVAILABLE:
                return self._fetch_earth2studio(lat, lon, date)
            else:
                return self._fetch_native(lat, lon, date)
        except Exception:
            logger.warning(
                "cBottle fetch failed for (%.4f, %.4f)", lat, lon, exc_info=True
            )
            return None

    def _fetch_earth2studio(
        self, lat: float, lon: float, date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Fetch using Earth2Studio API."""
        import numpy as np

        variables = ["rsds", "uas", "vas", "tas", "pr"]
        da = self._model([date], variables)

        # Extract nearest grid point to lat/lon
        # Earth2Studio returns xarray DataArray on HEALPix grid
        # Find nearest pixel
        result = {}
        for var in variables:
            if var in da.coords or hasattr(da, var):
                try:
                    val = float(da.sel(variable=var).values.flatten()[
                        _nearest_healpix_idx(lat, lon, da)
                    ])
                    result[var] = val
                except Exception:
                    continue

        return _convert_to_factor_kwargs(result)

    def _fetch_native(
        self, lat: float, lon: float, date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Fetch using native cBottle API."""
        import torch
        import numpy as np
        from cbottle.datasets.dataset_3d import get_dataset

        ds = get_dataset(dataset="amip")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        batch = next(iter(loader))

        output, coords = self._model.sample(batch)
        output = self._model.denormalize(output)

        # Extract variables at nearest grid point
        result = {}
        channel_names = coords.get("channel", [])
        for i, name in enumerate(channel_names):
            if name in ("rsds", "uas", "vas", "tas", "pr"):
                field = output[0, i].cpu().numpy()
                idx = _nearest_healpix_idx_from_field(lat, lon, field)
                result[name] = float(field.flatten()[idx])

        return _convert_to_factor_kwargs(result)


def _convert_to_factor_kwargs(raw: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Convert cBottle output variables to factor engine kwargs.

    Maps cBottle variable names to the same keys NASA POWER uses,
    so the factor engine doesn't need to know the data source.
    """
    if not raw:
        return None

    result: Dict[str, Any] = {}
    result["_source"] = "cbottle"

    # rsds (W/m2) → GHI (kWh/m2/day): multiply by daylight hours / 1000
    if "rsds" in raw:
        result["ghi_kwh_m2_day"] = raw["rsds"] * 12 / 1000  # ~12h avg daylight

    # uas + vas → wind speed, then scale 10m → 80m hub height
    if "uas" in raw and "vas" in raw:
        ws10 = math.sqrt(raw["uas"] ** 2 + raw["vas"] ** 2)
        result["wind_speed_ms"] = ws10 * (80 / 10) ** 0.14

    # tas (Kelvin in some outputs, Celsius in others)
    if "tas" in raw:
        temp = raw["tas"]
        if temp > 200:  # Kelvin
            temp -= 273.15
        result["avg_temp_c"] = temp

    # pr (kg/m2/s) → mm/day
    if "pr" in raw:
        result["precipitation_mm_day"] = raw["pr"] * 86400

    return result if len(result) > 1 else None


def _nearest_healpix_idx(lat: float, lon: float, da) -> int:
    """Find nearest HEALPix pixel index for a lat/lon in an xarray DataArray."""
    try:
        import healpy as hp
        nside = 64  # HEALPix-6
        theta = math.radians(90 - lat)
        phi = math.radians(lon % 360)
        return hp.ang2pix(nside, theta, phi, nest=True)
    except ImportError:
        # Fallback: just return center pixel
        return 0


def _nearest_healpix_idx_from_field(lat: float, lon: float, field) -> int:
    """Find nearest HEALPix pixel index from a numpy field."""
    try:
        import healpy as hp
        npix = field.size
        nside = hp.npix2nside(npix)
        theta = math.radians(90 - lat)
        phi = math.radians(lon % 360)
        return hp.ang2pix(nside, theta, phi, nest=True)
    except ImportError:
        return 0
