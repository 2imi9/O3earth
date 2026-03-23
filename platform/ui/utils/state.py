"""Session state management for the Streamlit app."""

import streamlit as st


DEFAULTS = {
    "selected_lat": 35.0,
    "selected_lon": -119.9,
    "selected_state": "CA",
    "selected_elevation": 0.0,
    "asset_type": "solar_utility",
    "capacity_mw": 100.0,
    "scenario": "SSP245",
    "target_year": 2050,
    "last_valuation": None,
    "last_climate_risk": None,
    "last_detection": None,
}


def init_state():
    """Initialize session state with defaults (idempotent)."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def update_location(lat: float, lon: float, state: str = ""):
    """Update selected location across all pages."""
    st.session_state.selected_lat = lat
    st.session_state.selected_lon = lon
    if state:
        st.session_state.selected_state = state
