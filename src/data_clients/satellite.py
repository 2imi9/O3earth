"""
Microsoft Planetary Computer Satellite Data Client.

Provides access to Sentinel-2 L2A imagery through the Planetary Computer
STAC API. Used to fetch satellite patches for OlmoEarth feature extraction.

Planetary Computer: https://planetarycomputer.microsoft.com/
STAC API: https://planetarycomputer.microsoft.com/api/stac/v1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel-2 band configuration matching OlmoEarth input requirements.
# OlmoEarth expects 12 bands in this order:
OLMOEARTH_BAND_ORDER = [
    "B02",  # Blue (10m)
    "B03",  # Green (10m)
    "B04",  # Red (10m)
    "B08",  # NIR (10m)
    "B05",  # Vegetation Red Edge 1 (20m)
    "B06",  # Vegetation Red Edge 2 (20m)
    "B07",  # Vegetation Red Edge 3 (20m)
    "B8A",  # Narrow NIR (20m)
    "B11",  # SWIR 1 (20m)
    "B12",  # SWIR 2 (20m)
    "B01",  # Coastal Aerosol (60m)
    "B09",  # Water Vapour (60m)
]

# Per-band normalization statistics for OlmoEarth.
# Normalization: (val - (mean - 2*std)) / (4*std)
# These values are from the OlmoEarth pretrain dataset.
OLMOEARTH_BAND_STATS = {
    "B02": {"mean": 1362.49, "std": 2520.77},
    "B03": {"mean": 1188.59, "std": 2399.30},
    "B04": {"mean": 1064.48, "std": 2658.56},
    "B08": {"mean": 1966.40, "std": 2847.11},
    "B05": {"mean": 1208.93, "std": 2556.32},
    "B06": {"mean": 1805.01, "std": 2820.81},
    "B07": {"mean": 1985.64, "std": 2944.84},
    "B8A": {"mean": 2139.06, "std": 2996.75},
    "B11": {"mean": 1620.97, "std": 2488.95},
    "B12": {"mean": 1085.46, "std": 2154.22},
    "B01": {"mean": 1347.46, "std": 2298.30},
    "B09": {"mean": 444.86, "std": 740.63},
}

# Planetary Computer STAC endpoints
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"


@dataclass
class SatellitePatch:
    """Container for a satellite image patch.

    Attributes:
        data: Pixel values as numpy array, shape [bands, height, width].
            None if not yet loaded.
        bands: List of band names in order.
        lat: Center latitude.
        lon: Center longitude.
        date: Acquisition date (ISO format string).
        crs: Coordinate reference system (e.g. "EPSG:32610").
        resolution_m: Pixel resolution in meters.
        cloud_cover_pct: Scene-level cloud cover percentage.
        stac_item_id: STAC item identifier for provenance.
    """

    data: Any  # np.ndarray or None
    bands: List[str]
    lat: float
    lon: float
    date: str
    crs: str
    resolution_m: float
    cloud_cover_pct: float
    stac_item_id: str


class PlanetaryComputerClient:
    """Client for fetching Sentinel-2 imagery from Microsoft Planetary Computer.

    This client handles STAC catalog search, asset signing, and data
    loading for Sentinel-2 L2A (atmospherically corrected) imagery.

    Usage::

        client = PlanetaryComputerClient()
        patch = client.get_sentinel2_patch(
            lat=35.0, lon=-120.0,
            date_range=("2023-06-01", "2023-09-01"),
            size_px=128
        )

    Note: This client requires the following optional dependencies:
        - pystac_client (STAC catalog search)
        - planetary_computer (asset signing)
        - rasterio / rioxarray (reading COG assets)
        - numpy
    """

    def __init__(self):
        """Initialize the Planetary Computer client.

        TODO:
            - Import and configure pystac_client.Client
            - Set up planetary_computer token signing
            - Configure connection pooling for parallel fetches
        """
        self._stac_client = None
        logger.info("PlanetaryComputerClient initialized (placeholder mode)")

    def _ensure_client(self):
        """Lazily initialize the STAC client.

        TODO:
            - import pystac_client
            - import planetary_computer
            - self._stac_client = pystac_client.Client.open(
                  PC_STAC_URL,
                  modifier=planetary_computer.sign_inplace
              )
        """
        if self._stac_client is not None:
            return

        logger.warning(
            "STAC client not initialized. Install pystac_client and "
            "planetary_computer packages to enable satellite data access."
        )

    def get_sentinel2_patch(
        self,
        lat: float,
        lon: float,
        date_range: Tuple[str, str],
        size_px: int = 128,
        max_cloud_cover: float = 20.0,
        bands: Optional[List[str]] = None,
    ) -> Optional[SatellitePatch]:
        """Fetch a Sentinel-2 L2A image patch for a given location.

        Searches the Planetary Computer STAC catalog for the least-cloudy
        Sentinel-2 scene covering the target location within the date range,
        then crops a (size_px x size_px) patch centered on (lat, lon).

        Args:
            lat: Center latitude (WGS84).
            lon: Center longitude (WGS84).
            date_range: Tuple of (start_date, end_date) in ISO format
                (e.g. ("2023-06-01", "2023-09-01")).
            size_px: Patch size in pixels. At 10m resolution, 128px = 1.28km.
                Default 128 (standard OlmoEarth input size).
            max_cloud_cover: Maximum acceptable cloud cover percentage.
                Default 20%.
            bands: List of band names to fetch. Default uses
                OLMOEARTH_BAND_ORDER (all 12 bands needed for OlmoEarth).

        Returns:
            SatellitePatch with image data, or None if no suitable scene
            is found.

        TODO:
            Implementation steps:
            1. Search STAC catalog:
               items = self._stac_client.search(
                   collections=[SENTINEL2_COLLECTION],
                   intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime=f"{date_range[0]}/{date_range[1]}",
                   query={"eo:cloud_cover": {"lt": max_cloud_cover}},
                   sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
                   max_items=5
               ).item_collection()

            2. Select best (least cloudy) item

            3. For each band, read signed COG asset:
               import rasterio
               href = planetary_computer.sign(item.assets[band].href)
               with rasterio.open(href) as src:
                   # Compute window centered on (lat, lon)
                   # Read size_px x size_px window
                   # Resample 20m/60m bands to 10m

            4. Stack bands into [12, size_px, size_px] numpy array

            5. Apply OlmoEarth normalization:
               for i, band in enumerate(bands):
                   stats = OLMOEARTH_BAND_STATS[band]
                   data[i] = (data[i] - (stats["mean"] - 2*stats["std"])) / (4*stats["std"])

            6. Return SatellitePatch
        """
        if bands is None:
            bands = OLMOEARTH_BAND_ORDER

        logger.warning(
            "get_sentinel2_patch is a placeholder. Returning None. "
            "Install pystac_client, planetary_computer, and rasterio "
            "to enable real satellite data access."
        )

        # Placeholder: return None until real implementation is added
        return None

    def search_scenes(
        self,
        lat: float,
        lon: float,
        date_range: Tuple[str, str],
        max_cloud_cover: float = 20.0,
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for available Sentinel-2 scenes at a location.

        Returns scene metadata without downloading pixel data. Useful
        for checking data availability before committing to a full fetch.

        Args:
            lat: Center latitude (WGS84).
            lon: Center longitude (WGS84).
            date_range: (start_date, end_date) in ISO format.
            max_cloud_cover: Maximum cloud cover percentage.
            max_items: Maximum number of scenes to return.

        Returns:
            List of scene metadata dicts with fields:
                - id: STAC item ID
                - datetime: acquisition datetime
                - cloud_cover: cloud cover percentage
                - platform: satellite (Sentinel-2A or 2B)

        TODO:
            - Implement STAC search (same as step 1 in get_sentinel2_patch)
            - Return metadata without reading pixel data
        """
        logger.warning(
            "search_scenes is a placeholder. Returning empty list."
        )
        return []

    @staticmethod
    def normalize_for_olmoearth(data, bands: Optional[List[str]] = None):
        """Apply OlmoEarth normalization to raw Sentinel-2 data.

        Normalization formula per band:
            normalized = (raw - (mean - 2*std)) / (4*std)

        Args:
            data: numpy array of shape [bands, H, W] with raw DN values.
            bands: Band names corresponding to data axis 0. Default
                uses OLMOEARTH_BAND_ORDER.

        Returns:
            Normalized numpy array, same shape as input.

        TODO:
            - import numpy as np
            - Apply per-band normalization using OLMOEARTH_BAND_STATS
        """
        if bands is None:
            bands = OLMOEARTH_BAND_ORDER

        logger.warning("normalize_for_olmoearth is a placeholder.")
        return data
