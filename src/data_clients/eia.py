"""
EIA (U.S. Energy Information Administration) API v2 Client.

Provides access to generator inventory, electricity prices, and plant
data from the EIA Open Data API. Used to fetch EIA Form 860 generator
data for training labels and NEMS economic data for valuation.

API Documentation: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# EIA API v2 base URL
EIA_API_BASE = "https://api.eia.gov/v2"

# Energy source code mapping (EIA codes -> our class labels)
EIA_FUEL_CODE_MAP = {
    "SUN": "solar",
    "WND": "wind",
    "NG": "gas",
    "COL": "coal",
    "NUC": "nuclear",
    "WAT": "hydro",
    "OIL": "oil",
    "WDS": "biomass",
    "BIT": "coal",        # Bituminous coal
    "SUB": "coal",        # Subbituminous coal
    "LIG": "coal",        # Lignite
    "PC": "oil",          # Petroleum coke
    "DFO": "oil",         # Distillate fuel oil
    "RFO": "oil",         # Residual fuel oil
    "GEO": "geothermal",
    "MWH": "storage",     # Battery / pumped hydro
    "WH": "other",        # Waste heat
    "BLQ": "biomass",     # Black liquor
    "OBG": "biomass",     # Other biomass gas
    "AB": "biomass",      # Agricultural byproduct
    "MSW": "biomass",     # Municipal solid waste
    "LFG": "biomass",     # Landfill gas
}


class EIAClient:
    """Client for the EIA Open Data API v2.

    Fetches generator inventory (Form 860), electricity prices, and
    plant-level data. Requires an API key set via the EIA_API_KEY
    environment variable or passed directly.

    Usage::

        client = EIAClient()
        generators = client.get_generators(state="CA", fuel_type="SUN")
        prices = client.get_electricity_prices(state="TX")
        plant = client.get_plant_by_id(56789)
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the EIA API client.

        Args:
            api_key: EIA API key. If not provided, reads from the
                EIA_API_KEY environment variable.

        Raises:
            ValueError: If no API key is available.
        """
        self.api_key = api_key or os.environ.get("EIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EIA API key required. Set EIA_API_KEY environment variable "
                "or pass api_key to the constructor."
            )
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated request to the EIA API v2.

        Args:
            endpoint: API endpoint path (e.g. "/electricity/operating-generator-capacity").
            params: Query parameters (api_key is added automatically).

        Returns:
            Parsed JSON response body.

        Raises:
            requests.HTTPError: On non-2xx response.
            requests.ConnectionError: On network failure.
        """
        url = f"{EIA_API_BASE}{endpoint}"
        if params is None:
            params = {}
        params["api_key"] = self.api_key

        logger.debug("EIA API request: GET %s params=%s", url, params)
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if "response" not in data:
            logger.warning("Unexpected EIA response format: %s", list(data.keys()))
        return data

    def get_generators(
        self,
        state: Optional[str] = None,
        fuel_type: Optional[str] = None,
        min_capacity_mw: Optional[float] = None,
        status: str = "operating",
        limit: int = 5000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch generator inventory data (EIA Form 860).

        Args:
            state: Two-letter US state code (e.g. "CA", "TX"). If None,
                returns all states.
            fuel_type: EIA energy source code (e.g. "SUN", "WND", "NG").
                See EIA_FUEL_CODE_MAP for valid codes.
            min_capacity_mw: Minimum nameplate capacity in MW. Filters
                out small generators.
            status: Generator status filter. Default "operating".
                Options: "operating", "proposed", "retired".
            limit: Maximum number of records per request (max 5000).
            offset: Pagination offset.

        Returns:
            List of generator records as dicts. Key fields:
                - plantCode: unique plant identifier
                - plantName: plant name
                - latitude, longitude: plant coordinates
                - entityName: owner/operator
                - state: US state
                - energySourceCode: fuel type code
                - nameplate_capacity_mw: installed capacity
                - operatingYearMonth: when generator began operating
        """
        params: Dict[str, Any] = {
            "frequency": "monthly",
            "data[0]": "nameplate-capacity-mw",
            "sort[0][column]": "nameplate-capacity-mw",
            "sort[0][direction]": "desc",
            "length": limit,
            "offset": offset,
        }

        # Build facet filters
        facet_idx = 0
        if state:
            params[f"facets[stateid][]"] = state.upper()
        if fuel_type:
            params[f"facets[energy_source_code][]"] = fuel_type.upper()
        if status:
            status_map = {
                "operating": "OP",
                "proposed": "P",
                "retired": "RE",
            }
            params[f"facets[status][]"] = status_map.get(status, status)

        try:
            data = self._request(
                "/electricity/operating-generator-capacity", params=params
            )
            records = data.get("response", {}).get("data", [])
            logger.info("EIA returned %d generator records", len(records))

            # Apply client-side capacity filter if needed
            if min_capacity_mw is not None:
                records = [
                    r for r in records
                    if _safe_float(r.get("nameplate-capacity-mw")) >= min_capacity_mw
                ]

            return records

        except requests.exceptions.RequestException:
            logger.exception("Failed to fetch EIA generators")
            return []

    def get_electricity_prices(
        self,
        state: Optional[str] = None,
        sector: str = "ALL",
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Fetch average retail electricity prices by state.

        Args:
            state: Two-letter US state code. If None, returns all states.
            sector: Customer sector filter. Options:
                "RES" (residential), "COM" (commercial), "IND" (industrial),
                "ALL" (all sectors).
            limit: Maximum records.

        Returns:
            List of price records. Key fields:
                - period: date period
                - stateid: state
                - sectorid: customer sector
                - price: cents per kWh
        """
        params: Dict[str, Any] = {
            "frequency": "monthly",
            "data[0]": "price",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": limit,
        }

        if state:
            params["facets[stateid][]"] = state.upper()
        if sector and sector != "ALL":
            params["facets[sectorid][]"] = sector

        try:
            data = self._request(
                "/electricity/retail-sales", params=params
            )
            records = data.get("response", {}).get("data", [])
            logger.info("EIA returned %d price records", len(records))
            return records

        except requests.exceptions.RequestException:
            logger.exception("Failed to fetch EIA electricity prices")
            return []

    def get_plant_by_id(
        self, plant_code: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch data for a specific power plant by its EIA plant code.

        Args:
            plant_code: EIA plant code (unique integer identifier from
                Form 860).

        Returns:
            Dict with plant data if found, None if not found or on error.
            Key fields: plantCode, plantName, latitude, longitude, state,
            entityName, generators (list), total capacity.
        """
        params: Dict[str, Any] = {
            "frequency": "monthly",
            "data[0]": "nameplate-capacity-mw",
            "facets[plantid][]": str(plant_code),
            "length": 100,
        }

        try:
            data = self._request(
                "/electricity/operating-generator-capacity", params=params
            )
            records = data.get("response", {}).get("data", [])
            if not records:
                logger.info("No records found for plant code %d", plant_code)
                return None

            # Aggregate generator data into a plant summary
            plant = {
                "plant_code": plant_code,
                "plant_name": records[0].get("plantName", "Unknown"),
                "state": records[0].get("stateid"),
                "latitude": _safe_float(records[0].get("latitude")),
                "longitude": _safe_float(records[0].get("longitude")),
                "entity_name": records[0].get("entityName"),
                "generators": records,
                "total_capacity_mw": sum(
                    _safe_float(r.get("nameplate-capacity-mw"))
                    for r in records
                ),
                "energy_sources": list(
                    set(r.get("energy_source_code", "") for r in records)
                ),
            }
            return plant

        except requests.exceptions.RequestException:
            logger.exception("Failed to fetch EIA plant %d", plant_code)
            return None

    def get_state_generation(
        self,
        state: str,
        fuel_type: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Fetch electricity generation data by state.

        Args:
            state: Two-letter US state code.
            fuel_type: EIA fuel type code to filter by (optional).
            limit: Maximum records.

        Returns:
            List of generation records with period, state, fuel type,
            and generation (MWh) fields.
        """
        params: Dict[str, Any] = {
            "frequency": "monthly",
            "data[0]": "generation",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "facets[stateid][]": state.upper(),
            "length": limit,
        }

        if fuel_type:
            params["facets[fueltypeid][]"] = fuel_type.upper()

        try:
            data = self._request(
                "/electricity/electric-power-operational-data", params=params
            )
            records = data.get("response", {}).get("data", [])
            logger.info(
                "EIA returned %d generation records for %s", len(records), state
            )
            return records

        except requests.exceptions.RequestException:
            logger.exception("Failed to fetch EIA generation for %s", state)
            return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
