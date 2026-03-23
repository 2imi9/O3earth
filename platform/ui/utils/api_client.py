"""HTTP client for the FastAPI backend."""

import os
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")


class EarthModelsAPI:
    """Thin HTTP client wrapping the FastAPI backend."""

    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health(self) -> dict | None:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(
        self,
        latitude: float,
        longitude: float,
        date_range: str = "2024-01-01/2024-12-31",
        max_cloud_cover: float = 10.0,
    ) -> dict | None:
        try:
            r = self.session.post(
                f"{self.base_url}/detect",
                json={
                    "latitude": latitude,
                    "longitude": longitude,
                    "date_range": date_range,
                    "max_cloud_cover": max_cloud_cover,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"Detection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Climate Risk
    # ------------------------------------------------------------------
    def assess_climate_risk(
        self,
        latitude: float,
        longitude: float,
        elevation: float = 0.0,
        asset_type: str = "solar",
        scenario: str = "SSP245",
        target_year: int = 2050,
    ) -> dict | None:
        try:
            r = self.session.post(
                f"{self.base_url}/climate-risk",
                json={
                    "latitude": latitude,
                    "longitude": longitude,
                    "elevation": elevation,
                    "asset_type": asset_type,
                    "scenario": scenario,
                    "target_year": target_year,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"Climate risk assessment failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------
    def value_asset(self, **kwargs) -> dict | None:
        try:
            r = self.session.post(
                f"{self.base_url}/value-asset",
                json=kwargs,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"Valuation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # EIA
    # ------------------------------------------------------------------
    def get_generators(self, state: str = None, energy_source: str = None, min_capacity_mw: float = 1.0) -> dict | None:
        try:
            params = {"min_capacity_mw": min_capacity_mw}
            if state:
                params["state"] = state
            if energy_source:
                params["energy_source"] = energy_source
            r = self.session.get(f"{self.base_url}/eia/generators", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def get_prices(self, scenario: str = "ref2025") -> dict | None:
        try:
            r = self.session.get(f"{self.base_url}/eia/prices", params={"scenario": scenario}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def get_state_summary(self, state: str) -> dict | None:
        try:
            r = self.session.get(f"{self.base_url}/eia/summary/{state}", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def llm_status(self) -> dict | None:
        try:
            r = self.session.get(f"{self.base_url}/llm/status", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def llm_chat(self, messages: list[dict], max_tokens: int = 8192, temperature: float = 1.0, enable_tools: bool = True, provider: str = "auto") -> dict | None:
        try:
            r = self.session.post(
                f"{self.base_url}/llm/chat",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "enable_tools": enable_tools,
                    "provider": provider,
                },
                timeout=300,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"LLM chat failed: {e}")
            return None

    def llm_analyze(self, analysis_type: str = "general", question: str = "", context: dict = None, provider: str = "auto") -> dict | None:
        try:
            r = self.session.post(
                f"{self.base_url}/llm/analyze",
                json={"analysis_type": analysis_type, "question": question, "context": context, "provider": provider},
                timeout=300,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"LLM analysis failed: {e}")
            return None


@st.cache_resource
def get_api_client() -> EarthModelsAPI:
    """Get or create a cached API client."""
    return EarthModelsAPI()
