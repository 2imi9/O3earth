"""Shared sidebar component."""

import streamlit as st
from ui.utils.api_client import get_api_client
from ui.utils.state import init_state


# Preset locations for quick selection
PRESETS = {
    "Custom": (None, None, ""),
    "Bakersfield, CA (Solar)": (35.37, -119.02, "CA"),
    "Amarillo, TX (Wind)": (35.22, -101.83, "TX"),
    "Topaz Solar Farm, CA": (35.05, -119.96, "CA"),
    "Alta Wind, CA": (35.07, -118.35, "CA"),
    "Horse Hollow, TX (Wind)": (32.07, -100.27, "TX"),
    "Grand Coulee Dam, WA": (47.95, -118.98, "WA"),
    "Ivanpah, NV (Solar)": (35.56, -115.47, "NV"),
    "Block Island, RI (Offshore Wind)": (41.17, -71.58, "RI"),
}


def render_sidebar():
    """Render the shared sidebar with location picker and status."""
    init_state()

    st.sidebar.title("O3 EartH")
    st.sidebar.caption("Geospatial Site Suitability Assessment")
    st.sidebar.divider()

    # API status
    api = get_api_client()
    health = api.health()
    if health:
        st.sidebar.success("API Connected")
        modules = health.get("modules", {})
        available = [k for k, v in modules.items() if v]
        st.sidebar.caption(f"Modules: {', '.join(available) if available else 'core only'}")
    else:
        st.sidebar.error("API Disconnected")
        st.sidebar.caption("Start the API: `uvicorn api.main:app`")

    st.sidebar.divider()

    # Quick location selector
    st.sidebar.subheader("Quick Location")
    preset = st.sidebar.selectbox("Preset Sites", list(PRESETS.keys()))
    if preset != "Custom":
        lat, lon, state = PRESETS[preset]
        st.session_state.selected_lat = lat
        st.session_state.selected_lon = lon
        if state:
            st.session_state.selected_state = state

    st.sidebar.number_input(
        "Latitude",
        value=st.session_state.selected_lat,
        min_value=-90.0,
        max_value=90.0,
        step=0.01,
        key="sidebar_lat",
        on_change=lambda: setattr(st.session_state, "selected_lat", st.session_state.sidebar_lat),
    )
    st.sidebar.number_input(
        "Longitude",
        value=st.session_state.selected_lon,
        min_value=-180.0,
        max_value=180.0,
        step=0.01,
        key="sidebar_lon",
        on_change=lambda: setattr(st.session_state, "selected_lon", st.session_state.sidebar_lon),
    )

    st.sidebar.divider()
    st.sidebar.caption("Northeastern University")
