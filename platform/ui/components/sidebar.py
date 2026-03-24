"""Shared sidebar component."""

import streamlit as st
from ui.utils.api_client import get_api_client
from ui.utils.state import init_state


# Preset locations for quick selection
PRESETS = {
    "Custom": (None, None, ""),
    "Mojave Desert, CA (Solar)": (35.0, -117.0, "CA"),
    "West Texas, US (Wind)": (31.95, -102.18, "TX"),
    "Kansas Plains, US (Wind)": (37.55, -98.24, "KS"),
    "Sahara, Morocco (Solar)": (31.63, -4.0, ""),
    "North Sea Coast, DE (Wind)": (54.0, 7.0, ""),
    "Patagonia, AR (Wind)": (-46.5, -69.0, ""),
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
